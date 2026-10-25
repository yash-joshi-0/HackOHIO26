//! Process-local exclusion between managed native inference and diagnostic profiling.
//!
//! Existing inference concurrency is preserved (read leases). Profiling takes an exclusive
//! lease only when idle. A lease follows the native child until it has been reaped, including
//! when its caller is cancelled. External processes and mobile in-process engines must join
//! this owner explicitly; this is not a device-wide or distributed reservation service.

use std::io;
use std::ops::{Deref, DerefMut};
use std::process::ExitStatus;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use tokio::process::{Child, Command};
use tokio::sync::{OwnedRwLockReadGuard, OwnedRwLockWriteGuard, RwLock};

#[derive(Clone, Default)]
pub struct ComputeAdmission {
    lock: Arc<RwLock<()>>,
    fenced: Arc<AtomicBool>,
}

impl ComputeAdmission {
    /// Shared by the managed process supervisor, shard stages and profiler in this process.
    pub fn process() -> Self {
        static ADMISSION: OnceLock<ComputeAdmission> = OnceLock::new();
        ADMISSION.get_or_init(Self::default).clone()
    }

    pub fn is_fenced(&self) -> bool {
        self.fenced.load(Ordering::Acquire)
    }

    /// No automatic reset: first verify that any unaccounted native work has stopped.
    pub fn fence(&self) {
        self.fenced.store(true, Ordering::Release);
    }

    pub async fn inference(&self) -> io::Result<InferenceLease> {
        self.check()?;
        let permit = Arc::clone(&self.lock).read_owned().await;
        self.check()?;
        Ok(InferenceLease {
            _permit: permit,
            admission: self.clone(),
        })
    }

    /// A diagnostic never queues ahead of inference. The caller may retry at an idle boundary.
    pub fn try_profile(&self) -> io::Result<ProfileLease> {
        self.check()?;
        let permit = Arc::clone(&self.lock)
            .try_write_owned()
            .map_err(|_| io::Error::new(io::ErrorKind::WouldBlock, "managed compute is busy"))?;
        self.check()?;
        Ok(ProfileLease {
            _permit: permit,
            admission: self.clone(),
            cleanup_confirmed: true,
        })
    }

    fn check(&self) -> io::Result<()> {
        if self.is_fenced() {
            return Err(io::Error::other(
                "compute fenced: native cleanup is unconfirmed",
            ));
        }
        Ok(())
    }
}

pub struct InferenceLease {
    _permit: OwnedRwLockReadGuard<()>,
    admission: ComputeAdmission,
}

pub struct ProfileLease {
    _permit: OwnedRwLockWriteGuard<()>,
    admission: ComputeAdmission,
    cleanup_confirmed: bool,
}

impl ProfileLease {
    pub fn execution_started(&mut self) {
        self.cleanup_confirmed = false;
    }

    pub fn confirm_cleanup(&mut self) {
        self.cleanup_confirmed = true;
    }
}

impl Drop for ProfileLease {
    fn drop(&mut self) {
        if !self.cleanup_confirmed {
            self.admission.fence();
        }
    }
}

/// A native child whose inference lease cannot be released before process exit.
pub struct AdmittedChild {
    child: Option<Child>,
    lease: Option<InferenceLease>,
}

impl AdmittedChild {
    pub async fn spawn(command: &mut Command) -> io::Result<Self> {
        Self::spawn_on(command, &ComputeAdmission::process()).await
    }

    pub async fn spawn_on(command: &mut Command, admission: &ComputeAdmission) -> io::Result<Self> {
        let lease = admission.inference().await?;
        let child = command.kill_on_drop(true).spawn()?;
        Ok(Self {
            child: Some(child),
            lease: Some(lease),
        })
    }

    pub async fn wait(&mut self) -> io::Result<ExitStatus> {
        let result = self.deref_mut().wait().await;
        if result.is_ok() {
            self.lease.take();
        }
        result
    }
}

impl Deref for AdmittedChild {
    type Target = Child;

    fn deref(&self) -> &Self::Target {
        self.child.as_ref().expect("child present until drop")
    }
}

impl DerefMut for AdmittedChild {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.child.as_mut().expect("child present until drop")
    }
}

impl Drop for AdmittedChild {
    fn drop(&mut self) {
        let Some(mut child) = self.child.take() else {
            return;
        };
        let Some(lease) = self.lease.take() else {
            return;
        };
        if matches!(child.try_wait(), Ok(Some(_))) {
            return;
        }
        let _ = child.start_kill();
        let mut cleanup = Cleanup {
            child,
            lease,
            confirmed: false,
        };
        match tokio::runtime::Handle::try_current() {
            Ok(runtime) => {
                runtime.spawn(async move {
                    cleanup.confirmed = cleanup.child.wait().await.is_ok();
                    drop(cleanup);
                });
            }
            Err(_) => drop(cleanup),
        }
    }
}

/// Fences before releasing the lease if the cleanup task itself is cancelled or panics.
struct Cleanup {
    child: Child,
    lease: InferenceLease,
    confirmed: bool,
}

impl Drop for Cleanup {
    fn drop(&mut self) {
        if !self.confirmed {
            // Runtime teardown may abort the async reaper. Confirm exit without depending on
            // that runtime before fencing. This bounded synchronous fallback runs only when
            // normal async cleanup did not finish; it never releases on elapsed time alone.
            let _ = self.child.start_kill();
            let deadline = Instant::now() + Duration::from_millis(100);
            loop {
                match self.child.try_wait() {
                    Ok(Some(_)) => return,
                    Ok(None) if Instant::now() < deadline => {
                        std::thread::sleep(Duration::from_millis(1));
                    }
                    _ => break,
                }
            }
            self.lease.admission.fence();
        }
    }
}

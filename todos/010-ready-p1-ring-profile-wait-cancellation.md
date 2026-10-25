---
status: ready
priority: p1
issue_id: "010"
tags: [runtime, cancellation, profiling]
dependencies: []
---

# Cancel a ring session while compute admission is held by profiling

M32 bounds writes but native spawn still awaited admission without watching the
upstream. A cancelled request could retain the ring gate and spawn after the
profile ended. This isolated follow-up is based on published PR52 head4a0bc7a.

- [x] Observe EOF/reset and stopped reception while awaiting native spawn.
- [x] Bound admission wait independently of write timeouts.
- [x] Prefer cancellation/deadline if a compute lease becomes available together.
- [x] Preserve existing child lifetime/reap admission and ring wire format.
- [x] Add socket-free queued-wait and cancellation-priority regressions.
- [x] Add real Iroh tests for finish, stop, deadline and successful next session.
- [ ] Independent loopback execution of the three new socket cases.
- [ ] Publication and regression validation on the reconciled main tree.

No hardware run is involved. Local socket execution remains blocked by sandbox
EPERM; network regressions are retained and must be run by publication.

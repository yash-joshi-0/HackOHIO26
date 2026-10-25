//! Bounded inspection of the existing export's metadata and physical tensor names.
//! No tensor decoding, model loading, or GGUF writing happens here.
use super::{Manifest, Role, seed::ExportReport};
use anyhow::{Context, Result, bail, ensure};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    io::{BufReader, Read, Seek, SeekFrom},
    path::Path,
};

const MAX_HEADER: u64 = 128 * 1024 * 1024;
struct Header {
    input: BufReader<File>,
}
impl Header {
    fn raw<const N: usize>(&mut self) -> Result<[u8; N]> {
        checked_header_end(self.input.stream_position()?, N as u64)?;
        let mut bytes = [0; N];
        self.input.read_exact(&mut bytes)?;
        Ok(bytes)
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.raw()?))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.raw()?))
    }
    fn skip(&mut self, bytes: u64) -> Result<()> {
        checked_header_end(self.input.stream_position()?, bytes)?;
        self.input.seek(SeekFrom::Current(bytes as i64))?;
        Ok(())
    }
    fn string(&mut self) -> Result<String> {
        let len = self.u64()?;
        ensure!(len <= 1024 * 1024, "GGUF string too long");
        checked_header_end(self.input.stream_position()?, len)?;
        let mut bytes = vec![0; len as usize];
        self.input.read_exact(&mut bytes)?;
        Ok(String::from_utf8(bytes)?)
    }
    fn skip_value(&mut self, kind: u32, depth: u32) -> Result<()> {
        ensure!(depth < 3, "GGUF array nesting too deep");
        match kind {
            0 | 1 | 7 => self.skip(1),
            2 | 3 => self.skip(2),
            4..=6 => self.skip(4),
            10..=12 => self.skip(8),
            8 => {
                let n = self.u64()?;
                self.skip(n)
            }
            9 => {
                let subtype = self.u32()?;
                let count = self.u64()?;
                ensure!(count <= 4_000_000, "GGUF array too large");
                for _ in 0..count {
                    self.skip_value(subtype, depth + 1)?;
                }
                Ok(())
            }
            _ => bail!("unknown GGUF metadata type"),
        }
    }
}

fn checked_header_end(position: u64, length: u64) -> Result<u64> {
    let end = position
        .checked_add(length)
        .context("GGUF header too large")?;
    ensure!(end <= MAX_HEADER, "GGUF header too large");
    Ok(end)
}

pub(super) fn check_export(path: &Path, m: &Manifest, report: &ExportReport) -> Result<()> {
    let mut h = Header {
        input: BufReader::new(File::open(path)?),
    };
    ensure!(
        &h.raw::<4>()? == b"GGUF" && h.u32()? == 3,
        "expected little-endian GGUF v3 export"
    );
    let tensors = h.u64()?;
    let fields = h.u64()?;
    ensure!(
        tensors > 0 && tensors <= 100_000 && fields <= 100_000,
        "invalid GGUF header counts"
    );
    let architecture_key = format!("{}.block_count", m.model.architecture);
    let mtp_key = format!("{}.nextn_predict_layers", m.model.architecture);
    let mut numbers = BTreeMap::new();
    let mut strings = BTreeMap::new();
    let mut field_names = BTreeSet::new();
    for _ in 0..fields {
        let key = h.string()?;
        let kind = h.u32()?;
        ensure!(
            field_names.insert(key.clone()),
            "duplicate GGUF metadata key"
        );
        if ["general.architecture", "meshthing.source_sha256"].contains(&key.as_str()) {
            ensure!(kind == 8, "incorrect GGUF string metadata type");
            strings.insert(key, h.string()?);
        } else if [
            architecture_key.as_str(),
            mtp_key.as_str(),
            "meshthing.shard_begin",
            "meshthing.shard_end",
        ]
        .contains(&key.as_str())
        {
            let n = match kind {
                4 => h.u32()? as u64,
                10 => h.u64()?,
                _ => bail!("incorrect GGUF integer metadata type"),
            };
            numbers.insert(key, n);
        } else {
            h.skip_value(kind, 0)?;
        }
    }
    ensure!(
        strings.get("general.architecture") == Some(&m.model.architecture)
            && strings.get("meshthing.source_sha256") == Some(&m.model.source_sha256),
        "GGUF source/architecture mismatch"
    );
    ensure!(
        numbers.get("meshthing.shard_begin") == Some(&(m.begin as u64))
            && numbers.get("meshthing.shard_end") == Some(&(m.end as u64)),
        "GGUF embedded range mismatch"
    );
    let blocks = *numbers
        .get(&architecture_key)
        .context("GGUF block count missing")?;
    let mtp = numbers.get(&mtp_key).copied().unwrap_or(0);
    ensure!(
        blocks.checked_sub(mtp) == Some(m.model.layers as u64),
        "GGUF decoder count mismatch"
    );
    let mut names = BTreeSet::new();
    let mut present_layers = BTreeSet::new();
    for _ in 0..tensors {
        let name = h.string()?;
        let dims = h.u32()?;
        ensure!((1..=4).contains(&dims), "invalid tensor dimensions");
        for _ in 0..dims {
            ensure!(h.u64()? > 0, "empty tensor");
        }
        h.u32()?; // Quantization type is preserved by the exporter, not decoded here.
        ensure!(h.u64()? < m.bytes, "tensor offset out of file");
        if let Some(rest) = name.strip_prefix("blk.") {
            let layer: u32 = rest
                .split('.')
                .next()
                .context("invalid block tensor")?
                .parse()?;
            ensure!(
                (m.begin..m.end).contains(&layer),
                "GGUF contains tensor outside advertised range"
            );
            present_layers.insert(layer);
        } else {
            let allowed = name.starts_with("rope_freqs.")
                || name.starts_with("token_embd.") && m.role != Role::Middle
                || name.starts_with("output.") && matches!(m.role, Role::Final | Role::Full)
                || name.starts_with("output_norm.") && matches!(m.role, Role::Final | Role::Full);
            ensure!(allowed, "unexpected role tensor {name}");
        }
        ensure!(names.insert(name), "duplicate GGUF tensor");
    }
    ensure!(
        present_layers == (m.begin..m.end).collect(),
        "GGUF missing advertised layers"
    );
    if matches!(m.role, Role::First | Role::Full) {
        ensure!(
            names.contains("token_embd.weight"),
            "first shard missing embeddings"
        );
    }
    if matches!(m.role, Role::Final | Role::Full) {
        ensure!(
            names.contains("output_norm.weight")
                && (names.contains("output.weight") || names.contains("token_embd.weight")),
            "final shard missing output tensors"
        );
    }
    let reported: BTreeSet<_> = report.tensors.iter().map(|t| t.name.clone()).collect();
    ensure!(
        reported.len() == report.tensors.len()
            && reported == names
            && report.tensors.iter().all(|t| t.bytes > 0),
        "export sidecar does not match physical GGUF tensors"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_limit_covers_string_payloads() {
        assert_eq!(checked_header_end(MAX_HEADER - 8, 8).unwrap(), MAX_HEADER);
        assert!(checked_header_end(MAX_HEADER - 7, 8).is_err());
        assert!(checked_header_end(u64::MAX, 1).is_err());
    }
}

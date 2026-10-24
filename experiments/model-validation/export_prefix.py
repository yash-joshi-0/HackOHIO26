#!/usr/bin/env python3
"""Copy a partition's quantized weights without dequantizing or renumbering layers."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf-python", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--cut", type=int, required=True)
    parser.add_argument("--begin", type=int, default=0)
    parser.add_argument("--expected-source-sha256")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    sys.path.insert(0, str(args.gguf_python))
    import gguf
    source_hash = digest(args.model)
    if args.expected_source_sha256 and source_hash != args.expected_source_sha256:
        raise ValueError("source checksum mismatch")
    reader = gguf.GGUFReader(args.model)
    architecture = reader.fields["general.architecture"].contents()
    layers = reader.fields[f"{architecture}.block_count"].contents()
    if architecture not in {"llama", "qwen2", "deepseek2"} or not 0 <= args.begin < args.cut <= layers:
        raise ValueError("invalid/unsupported partition")
    writer = gguf.GGUFWriter(args.output, architecture, endianess=reader.endianess)
    for field in reader.fields.values():
        if field.name == "general.architecture" or field.name.startswith("GGUF."):
            continue
        writer.add_key_value(field.name, field.contents(), field.types[0],
                             sub_type=field.types[-1] if field.types[0] == gguf.GGUFValueType.ARRAY else None)
    writer.add_string("meshthing.source_sha256", source_hash)
    writer.add_uint32("meshthing.shard_begin", args.begin)
    writer.add_uint32("meshthing.shard_end", args.cut)
    selected = []
    tied = not any(t.name == "output.weight" for t in reader.tensors)
    for tensor in reader.tensors:
        block = re.match(r"blk\.(\d+)\.", tensor.name)
        if block:
            keep = args.begin <= int(block[1]) < args.cut
        else:
            keep = (tensor.name.startswith("rope_freqs.") or
                    (tensor.name.startswith("token_embd.") and (args.begin == 0 or (args.cut == layers and tied))) or
                    (tensor.name.startswith(("output.", "output_norm.")) and args.cut == layers))
        if keep:
            selected.append(tensor)
            writer.add_tensor_info(tensor.name, tensor.data.shape, tensor.data.dtype,
                                   tensor.data.nbytes, tensor.tensor_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for tensor in selected:
        writer.write_tensor_data(tensor.data, tensor_endianess=reader.endianess)
    writer.close()
    report = {"source_sha256": source_hash, "artifact_sha256": digest(args.output),
              "begin": args.begin, "end": args.cut, "layers": layers, "architecture": architecture,
              "source_bytes": args.model.stat().st_size,
              "bytes": args.output.stat().st_size,
              "tensors": [{"name": t.name, "bytes": t.n_bytes} for t in selected]}
    args.output.with_suffix(args.output.suffix + ".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "tensors"}))


if __name__ == "__main__":
    main()

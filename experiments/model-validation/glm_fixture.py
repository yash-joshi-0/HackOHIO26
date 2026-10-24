#!/usr/bin/env python3
"""Build a two-layer numerical fixture from downloaded GLM tensor bytes."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf-python", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    sys.path.insert(0, str(args.gguf_python))
    import gguf

    class FixtureReader(gguf.GGUFReader):
        def _build_tensor_info(self, offset: int, count: int) -> tuple:
            end, fields = super()._build_tensor_info(offset, count)
            selected = []
            for field in fields:
                name = bytes(field.parts[1]).decode()
                block = re.match(r"blk\.(\d+)\.", name)
                if (block and int(block[1]) < 2) or not block:
                    selected.append(field)
            return end, selected

    reader = FixtureReader(args.source)
    architecture = reader.fields["general.architecture"].contents()
    if architecture != "deepseek2":
        raise ValueError("expected GLM's deepseek2 architecture")
    writer = gguf.GGUFWriter(args.output, architecture, endianess=reader.endianess)
    for field in reader.fields.values():
        if field.name == "general.architecture" or field.name.startswith("GGUF."):
            continue
        value = field.contents()
        if field.name == f"{architecture}.block_count":
            value = 2
        if "nextn_predict_layers" in field.name:
            value = 0
        writer.add_key_value(field.name, value, field.types[0],
                             sub_type=field.types[-1] if field.types[0] == gguf.GGUFValueType.ARRAY else None)
    for tensor in reader.tensors:
        writer.add_tensor_info(tensor.name, tensor.data.shape, tensor.data.dtype,
                               tensor.data.nbytes, tensor.tensor_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for tensor in reader.tensors:
        writer.write_tensor_data(tensor.data, tensor_endianess=reader.endianess)
    writer.close()
    digest = hashlib.sha256()
    with args.output.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    report = {"purpose": "truncated numerical fixture, not a usable GLM model",
              "layers": 2, "bytes": args.output.stat().st_size,
              "sha256": digest.hexdigest(), "source": str(args.source)}
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()

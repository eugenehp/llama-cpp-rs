#!/usr/bin/env python3
"""Filter prebuilt-matrix.json for workflow_dispatch inputs."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def main() -> int:
    matrix_path = Path(os.environ["MATRIX_JSON"])
    platform = os.environ.get("PLATFORM", "all")
    variant = os.environ.get("VARIANT", "all")
    library_type = os.environ.get("LIBRARY_TYPE", "both")
    event_name = os.environ.get("GITHUB_EVENT_NAME", "push")

    # Tag releases always publish the full matrix.
    if event_name == "push":
        platform = "all"
        variant = "all"
        library_type = "both"

    data = json.loads(matrix_path.read_text(encoding="utf-8"))

    def filter_entries(entries: list[dict]) -> list[dict]:
        out: list[dict] = []
        for entry in entries:
            suffix = entry["suffix"]
            if variant != "all" and suffix != variant:
                continue
            lt = entry["library_type"]
            if library_type == "both":
                pass
            elif library_type != lt:
                continue
            out.append(entry)
        return out

    outputs: dict[str, object] = {}
    for os_name in ("linux", "windows", "macos"):
        build = platform in ("all", os_name)
        filtered = filter_entries(data[os_name]) if build else []
        outputs[f"build_{os_name}"] = "true" if filtered else "false"
        outputs[f"{os_name}_matrix"] = json.dumps(filtered)

    github_output = os.environ.get("GITHUB_OUTPUT")
    if not github_output:
        print(json.dumps(outputs, indent=2))
        return 0

    with open(github_output, "a", encoding="utf-8") as fh:
        for key, value in outputs.items():
            fh.write(f"{key}={value}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

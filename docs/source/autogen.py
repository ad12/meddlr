"""Autogenerate certain assets/files for use with sphinx."""
import os
import pathlib
import re

import pandas as pd

_MEDDLR_PATH = pathlib.Path(os.path.join(os.path.dirname(__file__), "..", "..", "meddlr"))
_OUT_DIR = pathlib.Path(os.path.join(os.path.dirname(__file__), "..", "assets", "temp"))


def parse_config_to_csv(out_file: str):
    """Parse the meddlr config file into a csv for documentation."""
    with open(_MEDDLR_PATH / "config" / "defaults.py", "r") as f:
        lines = f.readlines()
    lines = [x.strip() for x in lines]

    docs = {"field": [], "description": []}
    comment = ""
    code = ""
    for line in lines:
        if line.startswith("#"):
            line = line.replace("#", "").strip()
            if re.match("^-+$", line):
                continue
            comment += f" {line}"
        elif line == "":
            comment = ""
            code = ""
        elif line.startswith("_C") and "=" in line:
            code = line.split("=")[0].strip().split(".", 1)[-1]
            docs["field"].append(code)
            docs["description"].append(comment.strip())
            comment = ""
            code = ""

    df = pd.DataFrame(docs)
    df = df.sort_values("field")
    df.to_csv(out_file, index=False)


def run():
    os.makedirs(_OUT_DIR, exist_ok=True)
    parse_config_to_csv(_OUT_DIR / "config-docs.csv")


if __name__ == "__main__":
    run()

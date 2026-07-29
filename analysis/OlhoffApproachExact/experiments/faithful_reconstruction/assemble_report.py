#!/usr/bin/env python3
"""Assemble faithful_reconstruction_report.md from its staged sections.

Run after analyze.py so that the generated tables are current.
"""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
SP = Path("/private/tmp/claude-502/-Users-piotrek-Programming-topOpt4freqMax/"
          "72f34ed9-f87e-4df1-a1c7-cc2bd8778148/scratchpad")
TABLES = (ROOT / "results" / "tables.md").read_text()


def block(header: str, until: str | None) -> str:
    i = TABLES.index(header)
    j = TABLES.index(until, i) if until else len(TABLES)
    return TABLES[i:j].rstrip()


parts = [SP / n for n in ("head.md", "sec5.md", "sec6.md", "sec6_addendum.md",
                          "sec7.md", "sec8.md", "sec9.md", "sec10.md",
                          "sec11.md", "rest.md")]
doc = "\n".join(p.read_text().rstrip("\n") for p in parts) + "\n"

gates = block("## Acceptance gates", None)
gates = gates.split("\n", 1)[1].strip()          # drop the duplicated header
doc = doc.replace("GATES_TABLE", gates)
doc = doc.replace("MESH_TRANSFER_SECTION", (SP / "mesh_transfer.md").read_text().strip())
doc = doc.replace("SCHEDULE_SENSITIVITY_TABLE", (SP / "sched_table.md").read_text().strip())
doc = doc.replace("SCHEDULE_SENSITIVITY_TEXT", (SP / "sched_text.md").read_text().strip())

while re.search(r"\n---\s*\n\s*\n---\s*\n", doc):        # collapse doubled separators
    doc = re.sub(r"\n---\s*\n\s*\n---\s*\n", "\n---\n\n", doc)

out = ROOT / "faithful_reconstruction_report.md"
out.write_text(doc)
n = len(doc.splitlines())
left = re.findall(r"\b(?:V5B_[A-Z0-9]+|[A-Z_]{6,}_TABLE|[A-Z_]{6,}_SECTION|[A-Z_]{6,}_TEXT)\b", doc)
print(f"wrote {out} ({n} lines)")
if left:
    print("UNFILLED PLACEHOLDERS:", sorted(set(left)))

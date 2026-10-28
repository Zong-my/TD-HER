"""Inspect the public tree without running experiments or producing artifacts."""
import ast
import hashlib
import re
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
allowed = {".py", ".md", ".txt", ".yaml", ".yml", ".json", ".raw", ".dyr"}
named = {".gitignore", ".gitkeep", "LICENSE", "SHA256SUMS"}
inputs = {"data/psse_cases/benchmark_ieee39/IEEE39.sav", "simulation/ieee300_gen_Hs.csv"}
failures = []
files = [p for p in ROOT.rglob("*") if p.is_file() and ".git" not in p.parts]
for path in files:
    relative = path.relative_to(ROOT).as_posix()
    if relative not in inputs and path.suffix.lower() not in allowed and path.name not in named:
        failures.append("Unexpected file type: " + relative)
    if any(x in path.relative_to(ROOT).parts for x in ("__pycache__", "checkpoints", ".pytest_cache")):
        failures.append("Generated directory: " + relative)
    if path.suffix == ".py":
        try:
            ast.parse(path.read_text(), filename=relative)
        except (SyntaxError, UnicodeError) as exc:
            failures.append(str(exc))
    if path.suffix in {".py", ".md", ".txt", ".yaml", ".json"}:
        text = path.read_text()
        # Private workstation/server paths have no role in portable sources.
        if re.search(r"/(?:home/myz|Users/myz|data/ssd)/", text):
            failures.append("Private absolute path: " + relative)
        if re.search(r"(?:gh[pousr]_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9]{40,}|BEGIN (?:RSA |OPENSSH )?PRIVATE KEY)", text):
            failures.append("Possible credential: " + relative)
for line in (ROOT/"data/psse_cases/SHA256SUMS").read_text().splitlines():
    digest, name = line.split("  ", 1)
    file = ROOT/"data/psse_cases"/name
    if hashlib.sha256(file.read_bytes()).hexdigest() != digest:
        failures.append("Input checksum mismatch: " + name)
for doc in [ROOT/"README.md", *ROOT.glob("docs/*.md")]:
    for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", doc.read_text()):
        if "://" not in target and not (doc.parent/target.split("#")[0]).exists():
            failures.append("Broken link in " + str(doc.relative_to(ROOT)) + ": " + target)
if failures:
    print("\n".join(failures))
    raise SystemExit(1)
print("PASS: source-only file types, syntax, private-path/credential scan, case checksums and local links.")
print("Files inspected:", len(files))

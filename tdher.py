"""Launch an experiment with data and generated outputs outside this checkout."""
import argparse
import os
from pathlib import Path
import runpy
import sys
from tdher_paths import REPO_ROOT, WORK_ROOT, DATA_ROOT, external_output

def _protect_sources(event, args):
    target = None
    if event == "open":
        path, mode, flags = args
        if isinstance(path, (str, bytes, os.PathLike)):
            writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (
                isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
            if writing:
                target = path
    elif event in ("os.mkdir", "os.remove", "os.rmdir", "os.truncate", "os.chmod"):
        target = args[0]
    elif event in ("os.rename", "os.link", "os.symlink"):
        for path in args[:2]:
            if isinstance(path, (str, bytes, os.PathLike)):
                external_output(os.fsdecode(path))
    if isinstance(target, (str, bytes, os.PathLike)):
        external_output(os.fsdecode(target))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("paths", help="Show configured external directories")
    sub.add_parser("list", help="List runnable experiment and preprocessing scripts")
    run = sub.add_parser("run", help="Run a source script in the external workspace")
    run.add_argument("script")
    run.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command == "paths":
        print("Source:", REPO_ROOT)
        print("Workspace:", WORK_ROOT)
        print("Data:", DATA_ROOT)
        return
    if args.command == "list":
        for directory in ("experiments", "data_proc", "analytical_comparators"):
            for file in sorted((REPO_ROOT / directory).rglob("*.py")):
                if '__name__' in file.read_text():
                    print(file.relative_to(REPO_ROOT))
        return
    script = (REPO_ROOT / args.script).resolve()
    if REPO_ROOT not in script.parents or not script.is_file() or script.suffix != ".py":
        parser.error("script must name a Python source file within this checkout")
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.chdir(WORK_ROOT)
    sys.path[:0] = [str(script.parent), str(REPO_ROOT)]
    sys.addaudithook(_protect_sources)
    tail = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    sys.argv = [str(script), *tail]
    runpy.run_path(str(script), run_name="__main__")

if __name__ == "__main__":
    main()

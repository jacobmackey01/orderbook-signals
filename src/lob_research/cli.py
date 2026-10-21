import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Reproducible LOBSTER order-book signal study")
    parser.add_argument("command", choices=["download", "run", "figures"])
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Repository root")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == "download":
        from .data import download_samples
        for row in download_samples(root):
            print(f"Verified {row['symbol']} {row['sha256'][:12]}")
    elif args.command == "run":
        from .pipeline import run
        run(root, args.config)
    else:
        from .presentation import build_figures, write_findings
        build_figures(root)
        write_findings(root)


if __name__ == "__main__":
    main()

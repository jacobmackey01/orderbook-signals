import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Reproducible LOBSTER order-book signal study")
    parser.add_argument("command", choices=["download", "run", "figures", "crypto-download", "crypto-run"])
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Repository root")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--phase", choices=["development", "heldout"], default="development",
                        help="crypto-run only: development folds or the one-off held-out folds")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == "download":
        from .data import download_samples
        for row in download_samples(root):
            print(f"Verified {row['symbol']} {row['sha256'][:12]}")
    elif args.command == "run":
        from .pipeline import run
        run(root, args.config)
    elif args.command == "crypto-download":
        import json
        from .crypto.tardis import download
        download(root, json.loads((args.config or root/"configs/crypto_binance.json").read_text()))
    elif args.command == "crypto-run":
        from .crypto.replication import run as crypto_run
        crypto_run(root, args.config, args.phase)
    else:
        from .presentation import build_figures, write_findings
        build_figures(root)
        write_findings(root)


if __name__ == "__main__":
    main()

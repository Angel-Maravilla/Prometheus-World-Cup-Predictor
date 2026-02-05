"""Entry point for `python -m wc_predictor`."""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="wc_predictor",
        description="World Cup match outcome prediction pipeline.",
    )
    parser.add_argument(
        "command",
        choices=["download", "features", "train", "evaluate", "predict"],
        help="Pipeline stage to run.",
    )
    # Only parse the first positional arg; the rest are forwarded.
    args, remaining = parser.parse_known_args()

    if args.command == "download":
        from wc_predictor.download_data import main as cmd
    elif args.command == "features":
        from wc_predictor.features import main as cmd
    elif args.command == "train":
        from wc_predictor.train import main as cmd
    elif args.command == "evaluate":
        from wc_predictor.evaluate import main as cmd
    elif args.command == "predict":
        from wc_predictor.predict import main as cmd
    else:
        parser.print_help()
        sys.exit(1)

    # Re-parse with the subcommand's own parser (via sys.argv manipulation)
    sys.argv = [f"wc_predictor.{args.command}"] + remaining
    cmd()


if __name__ == "__main__":
    main()

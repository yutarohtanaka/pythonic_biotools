import argparse
from .core import run_xenofilter


def main():
    parser = argparse.ArgumentParser(
        description="XenoFilter: Filter mouse (host) reads from human xenograft BAMs."
    )
    parser.add_argument("human_bam", help="Path to human-aligned BAM")
    parser.add_argument("mouse_bam", help="Path to mouse-aligned BAM")
    parser.add_argument("output_bam", help="Path to output filtered human BAM file")
    parser.add_argument(
        "--cutoff",
        type=float,
        default=4,
        help="Maximum per-read / mate-averaged edit distance cutoff (default: 4)",
    )
    parser.add_argument(
        "--unmapped-penalty",
        type=float,
        default=8,
        help="Unmapped mate penalty when --settings R_default is used (default: 8)",
    )
    parser.add_argument(
        "--settings",
        choices=["default", "R_default"],
        default="default",
        help="Parity mode: 'default' (Pythonic) or 'R_default' (upstream XenofilteR)",
    )

    args = parser.parse_args()
    run_xenofilter(
        args.human_bam,
        args.mouse_bam,
        args.output_bam,
        mismatch_cutoff=args.cutoff,
        unmapped_penalty=args.unmapped_penalty,
        settings=args.settings,
    )


if __name__ == "__main__":
    main()

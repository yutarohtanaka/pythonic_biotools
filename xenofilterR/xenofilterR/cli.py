import argparse
from .core import run_xenofilter

def main():
    parser = argparse.ArgumentParser(description="XenoFilter: Filter mouse reads from xenograft data.")
    parser.add_argument("human_bam", help="Path to human-aligned BAM")
    parser.add_argument("mouse_bam", help="Path to mouse-aligned BAM")
    parser.add_argument("output_bam", help="Path to output BAM file")
    parser.add_argument("--cutoff", type=int, default=5, help="Mismatch cutoff")

    args = parser.parse_args()
    run_xenofilter(args.human_bam, args.mouse_bam, args.output_bam, args.cutoff)


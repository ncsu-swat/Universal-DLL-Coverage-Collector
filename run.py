import argparse
from typing import Optional
import cov

def dll_choice(value: str) -> str:
    """Validate DLL name against supported choices."""
    choices = {"torch", "tf"}
    val = value.strip().lower()
    if val not in choices:
        raise argparse.ArgumentTypeError(
            f"Unsupported --dll '{value}'. Supported: {', '.join(sorted(choices))}"
        )
    return val


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Universal DLL Coverage Collector",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dll",
        type=dll_choice,
        required=True,
        help="Name of the DLL (choices: torch, tf)",
    )
    parser.add_argument(
        "--ver",
        type=str,
        required=True,
        help="Version of the DLL (e.g., 2.2.0, 2.2.0+cu121)",
    )
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="Target directory containing baseline generated outputs",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=False,
        default="_result",
        help="Output directory to write results",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        required=True,
        default=None,
        help="Baseline method",
    )
    parser.add_argument(
        "--itv",
        type=int,
        default=60,
        required=False,
        help="Interval in seconds for classifying Python files",
    )
    parser.add_argument(
        "--filter",
        type=str,
        required=False,
        default=None,
        help="Regex filter for classifying Python files",
    )
    parser.add_argument(
        "--num_parallel",
        type=int,
        default=16,
        required=False,
        help="Number of parallel workers for processing",
    )
    return parser



def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args


if __name__ == "__main__":
    parsed_args = parse_args()
    print(
        "Args:",
        {
            "dll": parsed_args.dll,
            "ver": parsed_args.ver,
            "target": str(parsed_args.target),
            "output": str(parsed_args.output),
            "baseline": str(parsed_args.baseline),
            "itv": parsed_args.itv,
            "filter": str(parsed_args.filter),
            "num_parallel": parsed_args.num_parallel,
        },
    )
    
    
    if parsed_args.dll == "torch":
        collector = cov.TorchCovCollector(
            ver=parsed_args.ver,
            target=parsed_args.target,
            output=parsed_args.output,
            baseline=parsed_args.baseline,
            dll=parsed_args.dll,
            itv=parsed_args.itv,
            filter=parsed_args.filter,
            num_parallel=parsed_args.num_parallel
        )
        collector.collect()

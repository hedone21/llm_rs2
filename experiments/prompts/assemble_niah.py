#!/usr/bin/env python3
"""NIAH 프롬프트 조합기.

benchmark_prompts.json의 filler_blocks와 needles를 사용하여
지정된 depth/length 조합의 NIAH 프롬프트를 생성한다.

Usage:
    python assemble_niah.py --needle N-PASS --depth 0.25 --blocks 4
    python assemble_niah.py --needle N-FACT --depth 0.5 --blocks 8
    python assemble_niah.py --all  # 모든 조합 생성 (stdout에 JSON)
"""

import argparse
import json
import sys
from pathlib import Path

PROMPTS_FILE = Path(__file__).parent / "benchmark_prompts.json"


def load_prompts():
    with open(PROMPTS_FILE, "r") as f:
        return json.load(f)


def assemble_niah_prompt(
    filler_blocks: list[dict],
    needle: dict,
    depth_ratio: float,
    num_blocks: int,
) -> str:
    """Assemble a NIAH prompt.

    Args:
        filler_blocks: List of filler text blocks.
        needle: Needle definition with needle_text and question.
        depth_ratio: Where to place the needle (0.0=start, 1.0=end).
        num_blocks: Total number of filler blocks to use.

    Returns:
        Assembled prompt string.
    """
    if num_blocks > len(filler_blocks):
        # Repeat filler blocks if needed
        repeats = (num_blocks // len(filler_blocks)) + 1
        available = (filler_blocks * repeats)[:num_blocks]
    else:
        available = filler_blocks[:num_blocks]

    # Calculate needle position
    needle_pos = max(0, min(num_blocks, int(depth_ratio * num_blocks)))

    # Split filler around needle
    before = available[:needle_pos]
    after = available[needle_pos:]

    # Assemble
    parts = []
    for block in before:
        parts.append(block["text"])
    parts.append(needle["needle_text"])
    for block in after:
        parts.append(block["text"])

    # Add question at the end
    parts.append(f"\n{needle['question']}")

    return "\n\n".join(parts)


def generate_all_combinations(data: dict) -> list[dict]:
    """Generate all NIAH prompt combinations."""
    niah = data["niah"]
    filler_blocks = niah["filler_blocks"]
    needles = niah["needles"]
    depths = niah["depth_ratios"]

    block_counts = [4, 6, 8]  # ~256, ~512, ~1024 토큰 추정
    combinations = []

    for needle in needles:
        for depth in depths:
            for num_blocks in block_counts:
                prompt = assemble_niah_prompt(
                    filler_blocks, needle, depth, num_blocks
                )
                combo_id = (
                    f"NIAH-{needle['id'].split('-')[1]}"
                    f"-D{int(depth*100)}"
                    f"-B{num_blocks}"
                )
                combinations.append({
                    "id": combo_id,
                    "needle_id": needle["id"],
                    "needle_type": needle["type"],
                    "depth_ratio": depth,
                    "num_blocks": num_blocks,
                    "expected_answer": needle["expected_answer"],
                    "prompt": prompt,
                })

    return combinations


def main():
    parser = argparse.ArgumentParser(description="NIAH prompt assembler")
    parser.add_argument("--needle", type=str, help="Needle ID (e.g., N-PASS)")
    parser.add_argument("--depth", type=float, help="Depth ratio (0.0 - 1.0)")
    parser.add_argument(
        "--blocks", type=int, default=4, help="Number of filler blocks"
    )
    parser.add_argument(
        "--all", action="store_true", help="Generate all combinations as JSON"
    )
    parser.add_argument(
        "--output", type=str, help="Output file (default: stdout)"
    )
    args = parser.parse_args()

    data = load_prompts()

    if args.all:
        combos = generate_all_combinations(data)
        output = json.dumps(combos, indent=2, ensure_ascii=False)
        if args.output:
            with open(args.output, "w") as f:
                f.write(output)
            print(f"Generated {len(combos)} combinations → {args.output}")
        else:
            print(output)
        return

    if not args.needle or args.depth is None:
        parser.error("--needle and --depth are required (or use --all)")

    niah = data["niah"]
    needle = next(
        (n for n in niah["needles"] if n["id"] == args.needle), None
    )
    if not needle:
        valid = [n["id"] for n in niah["needles"]]
        parser.error(f"Unknown needle: {args.needle}. Valid: {valid}")

    prompt = assemble_niah_prompt(
        niah["filler_blocks"], needle, args.depth, args.blocks
    )

    if args.output:
        with open(args.output, "w") as f:
            f.write(prompt)
        print(f"Prompt written to {args.output}", file=sys.stderr)
    else:
        print(prompt)


if __name__ == "__main__":
    main()

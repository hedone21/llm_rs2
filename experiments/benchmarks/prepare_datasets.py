#!/usr/bin/env python3
"""Download and format benchmark datasets for log-likelihood evaluation.

Downloads HellaSwag, ARC-Easy, and PIQA from HuggingFace datasets,
formats them with few-shot prompts, and outputs batch JSON files
ready for `generate --eval-ll --eval-batch`.

Usage:
    python experiments/benchmarks/prepare_datasets.py [--n-questions 100] [--n-shot 5]
"""

import argparse
import json
import os
import random

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def prepare_hellaswag(n_questions=100, n_shot=5, seed=42):
    """Prepare HellaSwag (sentence completion, 4-way)."""
    from datasets import load_dataset

    ds = load_dataset("Rowan/hellaswag", split="validation", trust_remote_code=False)

    random.seed(seed)
    indices = list(range(len(ds)))
    random.shuffle(indices)

    # Few-shot examples from first n_shot items
    few_shot_indices = indices[:n_shot]
    test_indices = indices[n_shot : n_shot + n_questions]

    def format_example(item, include_answer=True):
        ctx = item["ctx"]
        endings = item["endings"]
        label = int(item["label"])
        text = f"Context: {ctx}\n"
        for j, e in enumerate(endings):
            text += f"  {chr(65+j)}) {e}\n"
        if include_answer:
            text += f"Answer: {chr(65+label)}\n"
        return text, label, endings

    # Build few-shot prompt
    header = "Choose the most logical continuation.\n\n"
    few_shot_text = header
    for idx in few_shot_indices:
        ex_text, _, _ = format_example(ds[idx], include_answer=True)
        few_shot_text += ex_text + "\n"

    # Build evaluation tasks
    tasks = []
    for idx in test_indices:
        item = ds[idx]
        ctx = item["ctx"]
        endings = item["endings"]
        label = int(item["label"])

        question_text = f"Context: {ctx}\n"
        for j, e in enumerate(endings):
            question_text += f"  {chr(65+j)}) {e}\n"
        question_text += "Answer:"

        prompt = few_shot_text + question_text

        for j, ending in enumerate(endings):
            tasks.append(
                {
                    "id": f"hellaswag_{idx}_c{j}",
                    "question_id": f"hellaswag_{idx}",
                    "prompt": prompt,
                    "continuation": f" {ending}",
                    "choice_idx": j,
                    "gold": label,
                }
            )

    return tasks


def prepare_arc_easy(n_questions=100, n_shot=5, seed=42):
    """Prepare ARC-Easy (science QA, 4-way)."""
    from datasets import load_dataset

    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test", trust_remote_code=False)

    random.seed(seed + 1)
    indices = list(range(len(ds)))
    random.shuffle(indices)

    few_shot_indices = indices[:n_shot]
    test_indices = indices[n_shot : n_shot + n_questions]

    label_to_idx = {"A": 0, "B": 1, "C": 2, "D": 3, "1": 0, "2": 1, "3": 2, "4": 3}

    def format_example(item, include_answer=True):
        q = item["question"]
        choices = item["choices"]["text"]
        labels = item["choices"]["label"]
        answer_key = item["answerKey"]
        gold = label_to_idx.get(answer_key, 0)

        text = f"Question: {q}\n"
        for j, (lbl, c) in enumerate(zip(labels, choices)):
            text += f"  {lbl}) {c}\n"
        if include_answer:
            text += f"Answer: {answer_key}\n"
        return text, gold, choices, labels

    header = "Answer the following science questions.\n\n"
    few_shot_text = header
    for idx in few_shot_indices:
        ex_text, _, _, _ = format_example(ds[idx], include_answer=True)
        few_shot_text += ex_text + "\n"

    tasks = []
    for idx in test_indices:
        item = ds[idx]
        q = item["question"]
        choices = item["choices"]["text"]
        labels = item["choices"]["label"]
        answer_key = item["answerKey"]
        gold = label_to_idx.get(answer_key, 0)

        question_text = f"Question: {q}\n"
        for lbl, c in zip(labels, choices):
            question_text += f"  {lbl}) {c}\n"
        question_text += "Answer:"

        prompt = few_shot_text + question_text

        for j, (lbl, c) in enumerate(zip(labels, choices)):
            tasks.append(
                {
                    "id": f"arc_easy_{idx}_c{j}",
                    "question_id": f"arc_easy_{idx}",
                    "prompt": prompt,
                    "continuation": f" {c}",
                    "choice_idx": j,
                    "gold": gold,
                }
            )

    return tasks


def prepare_boolq(n_questions=100, n_shot=5, seed=42):
    """Prepare BoolQ (yes/no QA, 2-way)."""
    from datasets import load_dataset

    ds = load_dataset("google/boolq", split="validation")

    random.seed(seed + 2)
    indices = list(range(len(ds)))
    random.shuffle(indices)

    few_shot_indices = indices[:n_shot]
    test_indices = indices[n_shot : n_shot + n_questions]

    def format_example(item, include_answer=True):
        passage = item["passage"][:300]  # Truncate long passages
        question = item["question"]
        answer = item["answer"]  # bool
        label = 0 if answer else 1  # 0=Yes, 1=No

        text = f"Passage: {passage}\nQuestion: {question}?\nAnswer:"
        if include_answer:
            text += f" {'Yes' if answer else 'No'}\n"
        return text, label

    header = "Answer the following questions with Yes or No.\n\n"
    few_shot_text = header
    for idx in few_shot_indices:
        ex_text, _ = format_example(ds[idx], include_answer=True)
        few_shot_text += ex_text + "\n"

    tasks = []
    for idx in test_indices:
        item = ds[idx]
        passage = item["passage"][:300]
        question = item["question"]
        answer = item["answer"]
        label = 0 if answer else 1

        question_text = f"Passage: {passage}\nQuestion: {question}?\nAnswer:"
        prompt = few_shot_text + question_text

        for j, choice_label in enumerate(["Yes", "No"]):
            tasks.append(
                {
                    "id": f"boolq_{idx}_c{j}",
                    "question_id": f"boolq_{idx}",
                    "prompt": prompt,
                    "continuation": f" {choice_label}",
                    "choice_idx": j,
                    "gold": label,
                }
            )

    return tasks


def main():
    parser = argparse.ArgumentParser(description="Prepare benchmark datasets")
    parser.add_argument("--n-questions", type=int, default=100)
    parser.add_argument("--n-shot", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)

    benchmarks = {
        "hellaswag": prepare_hellaswag,
        "arc_easy": prepare_arc_easy,
        "boolq": prepare_boolq,
    }

    for name, prepare_fn in benchmarks.items():
        print(f"Preparing {name}...")
        tasks = prepare_fn(
            n_questions=args.n_questions, n_shot=args.n_shot, seed=args.seed
        )

        out_path = os.path.join(DATA_DIR, f"{name}.json")
        with open(out_path, "w") as f:
            json.dump(tasks, f, ensure_ascii=False)

        n_questions_actual = len(set(t["question_id"] for t in tasks))
        print(f"  {n_questions_actual} questions, {len(tasks)} eval tasks → {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()

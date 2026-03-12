#!/usr/bin/env python3
"""Download and format benchmark datasets for log-likelihood evaluation.

Supports H2O paper tasks (COPA, PiQA, Winogrande, OpenBookQA, RTE, MathQA)
and additional tasks (HellaSwag, ARC-Easy, BoolQ).

Usage:
    # H2O paper tasks only
    python experiments/benchmarks/prepare_datasets.py --tasks h2o

    # Specific tasks
    python experiments/benchmarks/prepare_datasets.py --tasks copa,piqa,winogrande

    # All tasks
    python experiments/benchmarks/prepare_datasets.py --tasks all
"""

import argparse
import json
import os
import random

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")


def _load_piqa_validation():
    """Download and parse PiQA validation set from raw files."""
    import io
    import zipfile
    import urllib.request

    cache_dir = os.path.join(RAW_DIR, "piqa")
    os.makedirs(cache_dir, exist_ok=True)

    goals_path = os.path.join(cache_dir, "dev.jsonl")
    labels_path = os.path.join(cache_dir, "dev-labels.lst")

    if not os.path.exists(goals_path):
        url = "https://storage.googleapis.com/ai2-mosaic/public/physicaliqa/physicaliqa-train-dev.zip"
        print(f"  Downloading PiQA from {url}...")
        resp = urllib.request.urlopen(url)
        zf = zipfile.ZipFile(io.BytesIO(resp.read()))
        for name in zf.namelist():
            if "dev" in name and not name.startswith("__"):
                basename = os.path.basename(name)
                if basename:
                    with open(os.path.join(cache_dir, basename), "wb") as f:
                        f.write(zf.read(name))

    with open(goals_path) as f:
        goals = [json.loads(line) for line in f]
    with open(labels_path) as f:
        labels = [int(line.strip()) for line in f]

    records = []
    for g, label in zip(goals, labels):
        records.append({
            "goal": g["goal"],
            "sol1": g["sol1"],
            "sol2": g["sol2"],
            "label": label,
        })
    return records


def _load_mathqa_test():
    """Download and parse MathQA test set from official zip."""
    import io
    import zipfile
    import urllib.request

    cache_dir = os.path.join(RAW_DIR, "mathqa")
    os.makedirs(cache_dir, exist_ok=True)
    test_path = os.path.join(cache_dir, "test.json")

    if not os.path.exists(test_path):
        url = "https://math-qa.github.io/math-QA/data/MathQA.zip"
        print(f"  Downloading MathQA from {url}...")
        resp = urllib.request.urlopen(url)
        zf = zipfile.ZipFile(io.BytesIO(resp.read()))
        for name in zf.namelist():
            basename = os.path.basename(name)
            if basename and basename.endswith(".json"):
                with open(os.path.join(cache_dir, basename), "wb") as f:
                    f.write(zf.read(name))

    with open(test_path) as f:
        data = json.load(f)

    return data


# H2O paper default n-shot per task (lm-eval-harness defaults)
DEFAULT_NSHOT = {
    "copa": 0,
    "piqa": 0,
    "winogrande": 5,
    "openbookqa": 0,
    "rte": 0,
    "mathqa": 0,
    "hellaswag": 10,
    "arc_easy": 0,
    "boolq": 0,
}

H2O_TASKS = ["copa", "piqa", "winogrande", "openbookqa", "rte", "mathqa"]


def prepare_copa(n_questions=100, n_shot=0, seed=42):
    """Prepare COPA (causal reasoning, 2-way). H2O paper: 0-shot."""
    from datasets import load_dataset

    ds = load_dataset("super_glue", "copa", split="validation")

    random.seed(seed)
    indices = list(range(len(ds)))
    random.shuffle(indices)

    few_shot_indices = indices[:n_shot] if n_shot > 0 else []
    test_indices = indices[n_shot : n_shot + n_questions]

    def format_example(item, include_answer=True):
        premise = item["premise"]
        connective = "because" if item["question"] == "cause" else "so"
        c1 = item["choice1"]
        c2 = item["choice2"]
        label = item["label"]  # 0 or 1

        text = f"Premise: {premise}\n"
        text += f"Question: What is the {item['question']}?\n"
        text += f"  A) {c1}\n  B) {c2}\n"
        if include_answer:
            text += f"Answer: {'A' if label == 0 else 'B'}\n"
        return text, label, [c1, c2]

    header = "Choose the more likely cause or effect.\n\n"
    few_shot_text = header
    for idx in few_shot_indices:
        ex_text, _, _ = format_example(ds[idx], include_answer=True)
        few_shot_text += ex_text + "\n"

    tasks = []
    for idx in test_indices:
        item = ds[idx]
        label = item["label"]
        choices = [item["choice1"], item["choice2"]]

        question_text = f"Premise: {item['premise']}\n"
        question_text += f"Question: What is the {item['question']}?\n"
        question_text += f"  A) {choices[0]}\n  B) {choices[1]}\n"
        question_text += "Answer:"

        prompt = few_shot_text + question_text

        for j, c in enumerate(choices):
            tasks.append({
                "id": f"copa_{idx}_c{j}",
                "question_id": f"copa_{idx}",
                "prompt": prompt,
                "continuation": f" {c}",
                "choice_idx": j,
                "gold": label,
            })

    return tasks


def prepare_piqa(n_questions=100, n_shot=0, seed=42):
    """Prepare PiQA (physical intuition QA, 2-way). H2O paper: 0-shot."""
    ds = _load_piqa_validation()

    random.seed(seed + 10)
    indices = list(range(len(ds)))
    random.shuffle(indices)

    few_shot_indices = indices[:n_shot] if n_shot > 0 else []
    test_indices = indices[n_shot : n_shot + n_questions]

    def format_example(item, include_answer=True):
        goal = item["goal"]
        s1 = item["sol1"]
        s2 = item["sol2"]
        label = item["label"]  # 0 or 1

        text = f"Goal: {goal}\n  A) {s1}\n  B) {s2}\n"
        if include_answer:
            text += f"Answer: {'A' if label == 0 else 'B'}\n"
        return text, label, [s1, s2]

    header = "Choose the most appropriate solution.\n\n"
    few_shot_text = header
    for idx in few_shot_indices:
        ex_text, _, _ = format_example(ds[idx], include_answer=True)
        few_shot_text += ex_text + "\n"

    tasks = []
    for idx in test_indices:
        item = ds[idx]
        label = item["label"]
        choices = [item["sol1"], item["sol2"]]

        question_text = f"Goal: {item['goal']}\n"
        question_text += f"  A) {choices[0]}\n  B) {choices[1]}\n"
        question_text += "Answer:"

        prompt = few_shot_text + question_text

        for j, c in enumerate(choices):
            tasks.append({
                "id": f"piqa_{idx}_c{j}",
                "question_id": f"piqa_{idx}",
                "prompt": prompt,
                "continuation": f" {c}",
                "choice_idx": j,
                "gold": label,
            })

    return tasks


def prepare_winogrande(n_questions=100, n_shot=5, seed=42):
    """Prepare Winogrande (pronoun resolution, 2-way). H2O paper: 5-shot."""
    from datasets import load_dataset

    ds = load_dataset("allenai/winogrande", "winogrande_xl", split="validation")

    random.seed(seed + 20)
    indices = list(range(len(ds)))
    random.shuffle(indices)

    few_shot_indices = indices[:n_shot] if n_shot > 0 else []
    test_indices = indices[n_shot : n_shot + n_questions]

    def format_example(item, include_answer=True):
        sentence = item["sentence"]
        opt1 = item["option1"]
        opt2 = item["option2"]
        label = int(item["answer"]) - 1  # "1" or "2" -> 0 or 1

        text = f"Sentence: {sentence}\n  A) {opt1}\n  B) {opt2}\n"
        if include_answer:
            text += f"Answer: {'A' if label == 0 else 'B'}\n"
        return text, label, [opt1, opt2]

    header = "Choose the correct word to fill the blank.\n\n"
    few_shot_text = header
    for idx in few_shot_indices:
        ex_text, _, _ = format_example(ds[idx], include_answer=True)
        few_shot_text += ex_text + "\n"

    tasks = []
    for idx in test_indices:
        item = ds[idx]
        label = int(item["answer"]) - 1
        choices = [item["option1"], item["option2"]]

        question_text = f"Sentence: {item['sentence']}\n"
        question_text += f"  A) {choices[0]}\n  B) {choices[1]}\n"
        question_text += "Answer:"

        prompt = few_shot_text + question_text

        for j, c in enumerate(choices):
            tasks.append({
                "id": f"winogrande_{idx}_c{j}",
                "question_id": f"winogrande_{idx}",
                "prompt": prompt,
                "continuation": f" {c}",
                "choice_idx": j,
                "gold": label,
            })

    return tasks


def prepare_openbookqa(n_questions=100, n_shot=0, seed=42):
    """Prepare OpenBookQA (science QA, 4-way). H2O paper: 0-shot."""
    from datasets import load_dataset

    ds = load_dataset("allenai/openbookqa", "main", split="test")

    random.seed(seed + 30)
    indices = list(range(len(ds)))
    random.shuffle(indices)

    few_shot_indices = indices[:n_shot] if n_shot > 0 else []
    test_indices = indices[n_shot : n_shot + n_questions]

    label_to_idx = {"A": 0, "B": 1, "C": 2, "D": 3}

    def format_example(item, include_answer=True):
        q = item["question_stem"]
        choices = item["choices"]["text"]
        labels = item["choices"]["label"]
        answer_key = item["answerKey"]
        gold = label_to_idx.get(answer_key, 0)

        text = f"Question: {q}\n"
        for lbl, c in zip(labels, choices):
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
        choices = item["choices"]["text"]
        labels = item["choices"]["label"]
        answer_key = item["answerKey"]
        gold = label_to_idx.get(answer_key, 0)

        question_text = f"Question: {item['question_stem']}\n"
        for lbl, c in zip(labels, choices):
            question_text += f"  {lbl}) {c}\n"
        question_text += "Answer:"

        prompt = few_shot_text + question_text

        for j, c in enumerate(choices):
            tasks.append({
                "id": f"openbookqa_{idx}_c{j}",
                "question_id": f"openbookqa_{idx}",
                "prompt": prompt,
                "continuation": f" {c}",
                "choice_idx": j,
                "gold": gold,
            })

    return tasks


def prepare_rte(n_questions=100, n_shot=0, seed=42):
    """Prepare RTE (textual entailment, 2-way). H2O paper: 0-shot."""
    from datasets import load_dataset

    ds = load_dataset("super_glue", "rte", split="validation")

    random.seed(seed + 40)
    indices = list(range(len(ds)))
    random.shuffle(indices)

    few_shot_indices = indices[:n_shot] if n_shot > 0 else []
    test_indices = indices[n_shot : n_shot + n_questions]

    def format_example(item, include_answer=True):
        premise = item["premise"]
        hypothesis = item["hypothesis"]
        label = item["label"]  # 0=entailment, 1=not_entailment

        text = f"Premise: {premise}\nHypothesis: {hypothesis}\n"
        text += "Does the premise entail the hypothesis?\n"
        if include_answer:
            text += f"Answer: {'Yes' if label == 0 else 'No'}\n"
        return text, label

    header = "Determine if the premise entails the hypothesis.\n\n"
    few_shot_text = header
    for idx in few_shot_indices:
        ex_text, _ = format_example(ds[idx], include_answer=True)
        few_shot_text += ex_text + "\n"

    choices_text = ["Yes", "No"]  # entailment=Yes, not_entailment=No
    tasks = []
    for idx in test_indices:
        item = ds[idx]
        label = item["label"]

        question_text = f"Premise: {item['premise']}\n"
        question_text += f"Hypothesis: {item['hypothesis']}\n"
        question_text += "Does the premise entail the hypothesis?\nAnswer:"

        prompt = few_shot_text + question_text

        for j, c in enumerate(choices_text):
            tasks.append({
                "id": f"rte_{idx}_c{j}",
                "question_id": f"rte_{idx}",
                "prompt": prompt,
                "continuation": f" {c}",
                "choice_idx": j,
                "gold": label,
            })

    return tasks


def prepare_mathqa(n_questions=100, n_shot=0, seed=42):
    """Prepare MathQA (math reasoning, 5-way). H2O paper: 0-shot."""
    ds = _load_mathqa_test()

    random.seed(seed + 50)
    indices = list(range(len(ds)))
    random.shuffle(indices)

    few_shot_indices = indices[:n_shot] if n_shot > 0 else []
    test_indices = indices[n_shot : n_shot + n_questions]

    label_map = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4}

    def parse_options(options_str):
        """Parse options string like 'a ) 1.5 , b ) 2.0 , ...'"""
        parts = []
        for opt in options_str.split(","):
            opt = opt.strip()
            if ")" in opt:
                _, text = opt.split(")", 1)
                parts.append(text.strip())
            elif opt:
                parts.append(opt.strip())
        return parts[:5]  # max 5 choices

    def format_example(item, include_answer=True):
        problem = item["Problem"]
        options = parse_options(item["options"])
        answer = label_map.get(item["correct"], 0)

        text = f"Problem: {problem}\n"
        for j, opt in enumerate(options):
            text += f"  {chr(65+j)}) {opt}\n"
        if include_answer:
            text += f"Answer: {chr(65+answer)}\n"
        return text, answer, options

    header = "Solve the following math problems.\n\n"
    few_shot_text = header
    for idx in few_shot_indices:
        ex_text, _, _ = format_example(ds[idx], include_answer=True)
        few_shot_text += ex_text + "\n"

    tasks = []
    for idx in test_indices:
        item = ds[idx]
        options = parse_options(item["options"])
        answer = label_map.get(item["correct"], 0)

        if len(options) < 2:
            continue

        question_text = f"Problem: {item['Problem']}\n"
        for j, opt in enumerate(options):
            question_text += f"  {chr(65+j)}) {opt}\n"
        question_text += "Answer:"

        prompt = few_shot_text + question_text

        for j, opt in enumerate(options):
            tasks.append({
                "id": f"mathqa_{idx}_c{j}",
                "question_id": f"mathqa_{idx}",
                "prompt": prompt,
                "continuation": f" {opt}",
                "choice_idx": j,
                "gold": answer,
            })

    return tasks


def prepare_hellaswag(n_questions=100, n_shot=10, seed=42):
    """Prepare HellaSwag (sentence completion, 4-way)."""
    from datasets import load_dataset

    ds = load_dataset("Rowan/hellaswag", split="validation", trust_remote_code=False)

    random.seed(seed)
    indices = list(range(len(ds)))
    random.shuffle(indices)

    few_shot_indices = indices[:n_shot] if n_shot > 0 else []
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

    header = "Choose the most logical continuation.\n\n"
    few_shot_text = header
    for idx in few_shot_indices:
        ex_text, _, _ = format_example(ds[idx], include_answer=True)
        few_shot_text += ex_text + "\n"

    tasks = []
    for idx in test_indices:
        item = ds[idx]
        endings = item["endings"]
        label = int(item["label"])

        question_text = f"Context: {item['ctx']}\n"
        for j, e in enumerate(endings):
            question_text += f"  {chr(65+j)}) {e}\n"
        question_text += "Answer:"

        prompt = few_shot_text + question_text

        for j, ending in enumerate(endings):
            tasks.append({
                "id": f"hellaswag_{idx}_c{j}",
                "question_id": f"hellaswag_{idx}",
                "prompt": prompt,
                "continuation": f" {ending}",
                "choice_idx": j,
                "gold": label,
            })

    return tasks


def prepare_arc_easy(n_questions=100, n_shot=0, seed=42):
    """Prepare ARC-Easy (science QA, 4-way)."""
    from datasets import load_dataset

    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test", trust_remote_code=False)

    random.seed(seed + 1)
    indices = list(range(len(ds)))
    random.shuffle(indices)

    few_shot_indices = indices[:n_shot] if n_shot > 0 else []
    test_indices = indices[n_shot : n_shot + n_questions]

    label_to_idx = {"A": 0, "B": 1, "C": 2, "D": 3, "1": 0, "2": 1, "3": 2, "4": 3}

    def format_example(item, include_answer=True):
        q = item["question"]
        choices = item["choices"]["text"]
        labels = item["choices"]["label"]
        answer_key = item["answerKey"]
        gold = label_to_idx.get(answer_key, 0)

        text = f"Question: {q}\n"
        for lbl, c in zip(labels, choices):
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
        choices = item["choices"]["text"]
        labels = item["choices"]["label"]
        answer_key = item["answerKey"]
        gold = label_to_idx.get(answer_key, 0)

        question_text = f"Question: {item['question']}\n"
        for lbl, c in zip(labels, choices):
            question_text += f"  {lbl}) {c}\n"
        question_text += "Answer:"

        prompt = few_shot_text + question_text

        for j, c in enumerate(choices):
            tasks.append({
                "id": f"arc_easy_{idx}_c{j}",
                "question_id": f"arc_easy_{idx}",
                "prompt": prompt,
                "continuation": f" {c}",
                "choice_idx": j,
                "gold": gold,
            })

    return tasks


def prepare_boolq(n_questions=100, n_shot=0, seed=42):
    """Prepare BoolQ (yes/no QA, 2-way)."""
    from datasets import load_dataset

    ds = load_dataset("google/boolq", split="validation")

    random.seed(seed + 2)
    indices = list(range(len(ds)))
    random.shuffle(indices)

    few_shot_indices = indices[:n_shot] if n_shot > 0 else []
    test_indices = indices[n_shot : n_shot + n_questions]

    def format_example(item, include_answer=True):
        passage = item["passage"][:300]
        question = item["question"]
        answer = item["answer"]
        label = 0 if answer else 1

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
            tasks.append({
                "id": f"boolq_{idx}_c{j}",
                "question_id": f"boolq_{idx}",
                "prompt": prompt,
                "continuation": f" {choice_label}",
                "choice_idx": j,
                "gold": label,
            })

    return tasks


ALL_BENCHMARKS = {
    "copa": prepare_copa,
    "piqa": prepare_piqa,
    "winogrande": prepare_winogrande,
    "openbookqa": prepare_openbookqa,
    "rte": prepare_rte,
    "mathqa": prepare_mathqa,
    "hellaswag": prepare_hellaswag,
    "arc_easy": prepare_arc_easy,
    "boolq": prepare_boolq,
}


def main():
    parser = argparse.ArgumentParser(description="Prepare benchmark datasets")
    parser.add_argument("--n-questions", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--tasks", default="h2o",
        help="Task set: 'h2o' (6 H2O paper tasks), 'all', or comma-separated names"
    )
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)

    if args.tasks == "h2o":
        task_names = H2O_TASKS
    elif args.tasks == "all":
        task_names = list(ALL_BENCHMARKS.keys())
    else:
        task_names = [t.strip() for t in args.tasks.split(",")]

    for name in task_names:
        if name not in ALL_BENCHMARKS:
            print(f"ERROR: Unknown task '{name}'. Available: {list(ALL_BENCHMARKS.keys())}")
            continue

        n_shot = DEFAULT_NSHOT.get(name, 0)
        prepare_fn = ALL_BENCHMARKS[name]

        print(f"Preparing {name} ({n_shot}-shot)...")
        tasks = prepare_fn(
            n_questions=args.n_questions, n_shot=n_shot, seed=args.seed
        )

        out_path = os.path.join(DATA_DIR, f"{name}.json")
        with open(out_path, "w") as f:
            json.dump(tasks, f, ensure_ascii=False)

        n_questions_actual = len(set(t["question_id"] for t in tasks))
        print(f"  {n_questions_actual} questions, {len(tasks)} eval tasks -> {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()

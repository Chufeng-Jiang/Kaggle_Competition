import re
import pandas as pd
from collections import Counter, defaultdict

from old.parser import detect_task_type


def extract_pairs(prompt):
    pairs = []
    for line in prompt.splitlines():
        line = line.strip()
        if " = " in line and not line.startswith("In "):
            x, y = line.split(" = ", 1)
            pairs.append((x.strip(), y.strip()))
    return pairs


def is_char_expr(x):
    x = str(x)
    if len(x) != 5:
        return False
    return not (x[:2].isdigit() and x[3:].isdigit())


def remove_controls(s):
    # 先只删除高置信 control symbols
    return "".join(ch for ch in s if ch not in {"+", "*"})


def align_by_length(src, tgt):
    """
    只分析长度相同的情况，避免乱配。
    """
    if len(src) != len(tgt):
        return []

    return list(zip(src, tgt))


def main():
    df = pd.read_csv("data/train.csv")

    raw_pair_counts = Counter()
    cleaned_pair_counts = Counter()

    src_counter = Counter()
    tgt_counter = Counter()

    same_len_raw = 0
    same_len_clean = 0
    total_char_pairs = 0

    examples_collected = defaultdict(list)

    for _, row in df.iterrows():
        prompt = row["prompt"]

        if detect_task_type(prompt) != "symbol":
            continue

        for x, y in extract_pairs(prompt):
            if not is_char_expr(x):
                continue

            total_char_pairs += 1

            x = str(x)
            y = str(y)

            # raw alignment
            pairs = align_by_length(x, y)
            if pairs:
                same_len_raw += 1
                for a, b in pairs:
                    raw_pair_counts[(a, b)] += 1
                    src_counter[a] += 1
                    tgt_counter[b] += 1
                    examples_collected[(a, b)].append((x, y, "raw"))

            # after deleting +/*
            cleaned = remove_controls(x)
            pairs = align_by_length(cleaned, y)
            if pairs:
                same_len_clean += 1
                for a, b in pairs:
                    cleaned_pair_counts[(a, b)] += 1
                    examples_collected[(a, b)].append((x, y, "cleaned"))

    rows = []

    for (a, b), cnt in cleaned_pair_counts.items():
        rows.append({
            "src": a,
            "tgt": b,
            "count": cnt,
            "type": "cleaned_delete_plus_star",
        })

    for (a, b), cnt in raw_pair_counts.items():
        rows.append({
            "src": a,
            "tgt": b,
            "count": cnt,
            "type": "raw",
        })

    out = pd.DataFrame(rows)
    out = out.sort_values("count", ascending=False)
    out.to_csv("outputs/char_substitution_stats.csv", index=False)

    print("total char pairs:", total_char_pairs)
    print("same length raw:", same_len_raw)
    print("same length after deleting +/*:", same_len_clean)

    print("\nTop raw substitutions:")
    raw = out[out["type"] == "raw"]
    print(raw.head(40).to_string(index=False))

    print("\nTop cleaned substitutions:")
    cleaned = out[out["type"] == "cleaned_delete_plus_star"]
    print(cleaned.head(40).to_string(index=False))

    print("\nTop source chars:")
    print(src_counter.most_common(30))

    print("\nTop target chars:")
    print(tgt_counter.most_common(30))

    print("\nExamples for top cleaned substitutions:")
    for _, r in cleaned.head(20).iterrows():
        key = (r["src"], r["tgt"])
        print("=" * 80)
        print(f"{r['src']} -> {r['tgt']} count={r['count']}")
        for x, y, typ in examples_collected[key][:10]:
            print(f"{x} -> {y} [{typ}]")

    print("\nsaved outputs/char_substitution_stats.csv")


if __name__ == "__main__":
    main()
    
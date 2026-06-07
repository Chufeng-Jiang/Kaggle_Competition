import re
import pandas as pd


MASK = 0xFF


def parse_bits(s):
    return int(s, 2)


def to_bits(x):
    return format(x & MASK, "08b")


def rol(x, k):
    k %= 8
    return ((x << k) | (x >> (8 - k))) & MASK


def ror(x, k):
    k %= 8
    return ((x >> k) | (x << (8 - k))) & MASK


def extract_bit_examples(prompt):
    return re.findall(r"([01]{8}) -> ([01]{8})", prompt)


def extract_bit_query(prompt):
    m = re.search(r"determine the output for: ([01]{8})", prompt)
    return m.group(1) if m else ""


def bit_candidates(x_bits):
    x = parse_bits(x_bits)
    c = {}

    c["identity"] = x
    c["not"] = (~x) & MASK

    for k in range(1, 8):
        c[f"shl_{k}"] = (x << k) & MASK
        c[f"shr_{k}"] = (x >> k) & MASK
        c[f"rol_{k}"] = rol(x, k)
        c[f"ror_{k}"] = ror(x, k)

    for k in range(256):
        c[f"xor_{k}"] = x ^ k
        c[f"and_{k}"] = x & k
        c[f"or_{k}"] = x | k

    for r in range(1, 8):
        rx = rol(x, r)
        for k in range(256):
            c[f"rol_{r}_xor_{k}"] = rx ^ k

        rx = ror(x, r)
        for k in range(256):
            c[f"ror_{r}_xor_{k}"] = rx ^ k

    return {name: to_bits(v) for name, v in c.items()}


def score_rules(examples):
    scores = {}

    for x, y in examples:
        for name, val in bit_candidates(x).items():
            if val == y:
                scores[name] = scores.get(name, 0) + 1

    return scores


def choose_rule(scores):
    if not scores:
        return None

    best = max(scores.values())
    possible = [k for k, v in scores.items() if v == best]

    def complexity(name):
        if name == "identity":
            return 0
        if name == "not":
            return 1
        if name.startswith(("rol_", "ror_", "shl_", "shr_")) and "_xor_" not in name:
            return 2
        if name.startswith("xor_"):
            return 3
        if name.startswith("and_") or name.startswith("or_"):
            return 4
        if "_xor_" in name:
            return 5
        return 99

    possible.sort(key=lambda r: (complexity(r), r))
    return possible[0]


def solve_bit_prompt_byte(prompt):
    examples = extract_bit_examples(prompt)
    query = extract_bit_query(prompt)

    if not examples or not query:
        return "", None, 0

    scores = score_rules(examples)
    rule = choose_rule(scores)

    if rule is None:
        return "", None, 0

    pred = bit_candidates(query).get(rule, "")
    return pred, rule, scores[rule]


def main():
    df = pd.read_csv("data/train.csv")
    bit_df = df[df["prompt"].str.contains("8-bit binary", case=False)].copy()

    rows = []

    for _, row in bit_df.iterrows():
        pred, rule, score = solve_bit_prompt_byte(row["prompt"])
        rows.append({
            "id": row["id"],
            "pred": pred,
            "answer": row["answer"],
            "correct": str(pred) == str(row["answer"]),
            "rule": rule,
            "score": score,
        })

    out = pd.DataFrame(rows)
    out.to_csv("outputs/bit_search_results.csv", index=False)

    print("bit tasks:", len(out))
    print("bit accuracy:", out["correct"].mean())
    print("\nTop rules:")
    print(out["rule"].value_counts(dropna=False).head(20))
    print("\nsaved outputs/bit_search_results.csv")


if __name__ == "__main__":
    main()
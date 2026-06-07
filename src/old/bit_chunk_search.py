import re
import itertools
import pandas as pd


def extract_bit_examples(prompt):
    return re.findall(r"([01]{8}) -> ([01]{1,8})", prompt)


def extract_bit_query(prompt):
    m = re.search(r"determine the output for: ([01]{8})", prompt)
    return m.group(1) if m else ""


def normalize_bits(s):
    s = str(s)
    if len(s) < 8:
        return s.zfill(8)
    return s


def infer_output_format(examples):
    outputs = [y for _, y in examples]
    padded = sum(len(y) == 8 for y in outputs)
    unpadded = sum(len(y) < 8 for y in outputs)

    if unpadded > padded:
        return "unpadded"

    return "padded"


def apply_format(bits, fmt):
    if fmt == "unpadded":
        return bits.lstrip("0") or "0"
    return bits


def split_chunks(bits, size):
    return [
        bits[i:i + size]
        for i in range(0, 8, size)
    ]


def join_chunks(chunks):
    return "".join(chunks)


def chunk_xor(ch, k):
    size = len(ch)
    v = int(ch, 2)
    return format(v ^ k, f"0{size}b")


def chunk_add(ch, k):
    size = len(ch)
    mod = 1 << size
    v = int(ch, 2)
    return format((v + k) % mod, f"0{size}b")


def chunk_sub(ch, k):
    size = len(ch)
    mod = 1 << size
    v = int(ch, 2)
    return format((v - k) % mod, f"0{size}b")


def rotate_list(xs, k):
    if not xs:
        return xs
    k %= len(xs)
    return xs[k:] + xs[:k]


def chunk_candidates(x_bits):
    bits = normalize_bits(x_bits)
    out = {}

    for size in [1, 2, 4]:
        chunks = split_chunks(bits, size)
        n = len(chunks)

        # identity
        out[f"chunk{size}_identity"] = join_chunks(chunks)

        # reverse chunk order
        out[f"chunk{size}_reverse_chunks"] = join_chunks(list(reversed(chunks)))

        # reverse bits inside each chunk
        out[f"chunk{size}_reverse_each"] = join_chunks([c[::-1] for c in chunks])

        # reverse chunks + reverse each
        out[f"chunk{size}_reverse_both"] = join_chunks(
            [c[::-1] for c in reversed(chunks)]
        )

        # rotate chunk order
        for k in range(1, n):
            out[f"chunk{size}_rotl_{k}"] = join_chunks(rotate_list(chunks, k))
            out[f"chunk{size}_rotr_{k}"] = join_chunks(rotate_list(chunks, -k))

        # chunk permutation
        if n <= 4:
            for perm in itertools.permutations(range(n)):
                name = f"chunk{size}_perm_" + "_".join(map(str, perm))
                out[name] = join_chunks([chunks[i] for i in perm])

        max_const = 1 << size

        # apply same xor/add/sub to every chunk
        for k in range(max_const):
            out[f"chunk{size}_xor_each_{k}"] = join_chunks(
                [chunk_xor(c, k) for c in chunks]
            )

            out[f"chunk{size}_add_each_{k}"] = join_chunks(
                [chunk_add(c, k) for c in chunks]
            )

            out[f"chunk{size}_sub_each_{k}"] = join_chunks(
                [chunk_sub(c, k) for c in chunks]
            )

        # rotate then xor/add/sub
        for r in range(1, n):
            rot = rotate_list(chunks, r)

            for k in range(max_const):
                out[f"chunk{size}_rotl_{r}_xor_each_{k}"] = join_chunks(
                    [chunk_xor(c, k) for c in rot]
                )

                out[f"chunk{size}_rotl_{r}_add_each_{k}"] = join_chunks(
                    [chunk_add(c, k) for c in rot]
                )

                out[f"chunk{size}_rotl_{r}_sub_each_{k}"] = join_chunks(
                    [chunk_sub(c, k) for c in rot]
                )

            rot = rotate_list(chunks, -r)

            for k in range(max_const):
                out[f"chunk{size}_rotr_{r}_xor_each_{k}"] = join_chunks(
                    [chunk_xor(c, k) for c in rot]
                )

                out[f"chunk{size}_rotr_{r}_add_each_{k}"] = join_chunks(
                    [chunk_add(c, k) for c in rot]
                )

                out[f"chunk{size}_rotr_{r}_sub_each_{k}"] = join_chunks(
                    [chunk_sub(c, k) for c in rot]
                )

    return out


def score_rules(examples):
    scores = {}

    for x, y in examples:
        y_norm = normalize_bits(y)
        cands = chunk_candidates(x)

        for rule, pred in cands.items():
            if pred == y_norm:
                scores[rule] = scores.get(rule, 0) + 1

    return scores


def rule_complexity(rule):
    if "identity" in rule:
        return 0
    if "rot" in rule:
        return 1
    if "reverse_chunks" in rule:
        return 2
    if "perm" in rule:
        return 3
    if "xor_each" in rule:
        return 4
    if "add_each" in rule or "sub_each" in rule:
        return 5
    return 10


def choose_rule(scores):
    if not scores:
        return None

    best = max(scores.values())
    rules = [r for r, s in scores.items() if s == best]

    rules.sort(key=lambda r: (rule_complexity(r), r))
    return rules[0]


def solve_bit_prompt_chunk(prompt):
    examples = extract_bit_examples(prompt)
    query = extract_bit_query(prompt)

    if not examples or not query:
        return "", None, 0

    fmt = infer_output_format(examples)

    scores = score_rules(examples)
    rule = choose_rule(scores)

    if rule is None:
        return "", None, 0

    pred = chunk_candidates(query).get(rule, "")
    pred = apply_format(pred, fmt)

    return pred, rule, scores[rule]


def main():
    df = pd.read_csv("data/train.csv")
    bit_df = df[df["prompt"].str.contains("8-bit binary", case=False)].copy()

    rows = []

    for _, row in bit_df.iterrows():
        pred, rule, score = solve_bit_prompt_chunk(row["prompt"])

        rows.append({
            "id": row["id"],
            "pred": pred,
            "answer": str(row["answer"]),
            "correct": str(pred) == str(row["answer"]),
            "rule": rule,
            "score": score,
        })

    out = pd.DataFrame(rows)
    out.to_csv("outputs/bit_chunk_search_results.csv", index=False)

    print("bit tasks:", len(out))
    print("chunk accuracy:", out["correct"].mean())
    print("\nTop rules:")
    print(out["rule"].value_counts(dropna=False).head(30))

    print("\nCorrect examples:")
    print(out[out["correct"] == True].head(30).to_string(index=False))

    print("\nWrong examples:")
    print(out[out["correct"] == False].head(30).to_string(index=False))

    print("\nsaved outputs/bit_chunk_search_results.csv")


if __name__ == "__main__":
    main()
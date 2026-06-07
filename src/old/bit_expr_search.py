import re
import pandas as pd



def bits_to_list(s: str):
    return [int(c) for c in s.strip()]


def list_to_bits(xs):
    return "".join(str(int(x)) for x in xs)


def majority(a, b, c):
    return int(a + b + c >= 2)


def choice(a, b, c):
    return b if a else c

def count_ones(x):
    return sum(x)


def count_zeros(x):
    return 8 - sum(x)


def has_run(x, bit, length):
    target = [bit] * length

    for i in range(0, 8 - length + 1):
        if x[i:i + length] == target:
            return 1

    return 0


def window_all(x, start, length, bit):
    if start + length > 8:
        return 0

    for i in range(start, start + length):
        if x[i] != bit:
            return 0

    return 1


def window_parity(x, start, length):
    if start + length > 8:
        return 0

    return sum(x[start:start + length]) % 2


def window_majority_one(x, start, length):
    if start + length > 8:
        return 0

    return int(sum(x[start:start + length]) >= (length + 1) // 2)


def extract_bit_examples(prompt: str):
    return re.findall(r"([01]{8}) -> ([01]{8})", prompt)


def extract_bit_query(prompt: str):
    m = re.search(r"determine the output for: ([01]{8})", prompt)
    return m.group(1) if m else ""


def build_expr_library():
    exprs = []

    exprs.append(("const_0", lambda x: 0))
    exprs.append(("const_1", lambda x: 1))

    for i in range(8):
        exprs.append((f"x{i}", lambda x, i=i: x[i]))
        exprs.append((f"not_x{i}", lambda x, i=i: 1 - x[i]))

    for i in range(8):
        for j in range(8):
            exprs.append((f"x{i}_xor_x{j}", lambda x, i=i, j=j: x[i] ^ x[j]))
            exprs.append((f"x{i}_and_x{j}", lambda x, i=i, j=j: x[i] & x[j]))
            exprs.append((f"x{i}_or_x{j}", lambda x, i=i, j=j: x[i] | x[j]))

    for i in range(8):
        for j in range(8):
            for k in range(8):
                exprs.append(
                    (
                        f"maj_{i}_{j}_{k}",
                        lambda x, i=i, j=j, k=k: majority(x[i], x[j], x[k]),
                    )
                )
                exprs.append(
                    (
                        f"ch_{i}_{j}_{k}",
                        lambda x, i=i, j=j, k=k: choice(x[i], x[j], x[k]),
                    )
                )
    
    # -------------------------
    # sequence / pattern features
    # -------------------------

    # count threshold features
    for k in range(0, 9):
        exprs.append((
            f"count1_eq_{k}",
            lambda x, k=k: int(count_ones(x) == k)
        ))

        exprs.append((
            f"count1_ge_{k}",
            lambda x, k=k: int(count_ones(x) >= k)
        ))

        exprs.append((
            f"count1_le_{k}",
            lambda x, k=k: int(count_ones(x) <= k)
        ))

        exprs.append((
            f"count0_eq_{k}",
            lambda x, k=k: int(count_zeros(x) == k)
        ))

        exprs.append((
            f"count0_ge_{k}",
            lambda x, k=k: int(count_zeros(x) >= k)
        ))

        exprs.append((
            f"count0_le_{k}",
            lambda x, k=k: int(count_zeros(x) <= k)
        ))

    # existence of consecutive runs
    for length in range(2, 9):
        exprs.append((
            f"has_run1_len_{length}",
            lambda x, length=length: has_run(x, 1, length)
        ))

        exprs.append((
            f"has_run0_len_{length}",
            lambda x, length=length: has_run(x, 0, length)
        ))

    # local sliding windows
    for length in range(2, 9):
        for start in range(0, 8 - length + 1):

            exprs.append((
                f"win_{start}_{length}_all1",
                lambda x, start=start, length=length:
                    window_all(x, start, length, 1)
            ))

            exprs.append((
                f"win_{start}_{length}_all0",
                lambda x, start=start, length=length:
                    window_all(x, start, length, 0)
            ))

            exprs.append((
                f"win_{start}_{length}_parity",
                lambda x, start=start, length=length:
                    window_parity(x, start, length)
            ))

            exprs.append((
                f"win_{start}_{length}_maj1",
                lambda x, start=start, length=length:
                    window_majority_one(x, start, length)
            ))

    return exprs


EXPR_LIBRARY = build_expr_library()


def expr_complexity(name: str):
    if name.startswith("const"):
        return 5
    if name.startswith("not_x"):
        return 2
    if re.fullmatch(r"x\d", name):
        return 1
    if "_xor_" in name:
        return 3
    if "_and_" in name:
        return 4
    if "_or_" in name:
        return 4
    if name.startswith("maj_"):
        return 5
    if name.startswith("ch_"):
        return 6
    
    if name.startswith("count"):
        return 7

    if name.startswith("has_run"):
        return 7

    if name.startswith("win_"):
        return 8
    return 99


def learn_one_output_bit(examples, output_idx):
    best_name = None
    best_fn = None
    best_score = -1
    best_complexity = 999

    for name, fn in EXPR_LIBRARY:
        score = 0

        for x_bits, y_bits in examples:
            x = bits_to_list(x_bits)
            y = int(y_bits[output_idx])
            pred = fn(x)

            if pred == y:
                score += 1

        comp = expr_complexity(name)

        if (
            score > best_score
            or (score == best_score and comp < best_complexity)
            or (
                score == best_score
                and comp == best_complexity
                and str(name) < str(best_name)
            )
        ):
            best_name = name
            best_fn = fn
            best_score = score
            best_complexity = comp

    return best_name, best_fn, best_score


def learn_rule(examples):
    rule = []

    for out_idx in range(8):
        name, fn, score = learn_one_output_bit(examples, out_idx)
        rule.append({
            "out_idx": out_idx,
            "name": name,
            "fn": fn,
            "score": score,
        })

    return rule


def predict(rule, x_bits):
    x = bits_to_list(x_bits)
    return list_to_bits([item["fn"](x) for item in rule])


def rule_train_accuracy(rule, examples):
    correct = 0

    for x_bits, y_bits in examples:
        if predict(rule, x_bits) == y_bits:
            correct += 1

    return correct / len(examples)


def infer_bit_output_format(examples):
    outputs = [y for _, y in examples]

    padded_count = sum(len(y) == 8 for y in outputs)
    unpadded_count = sum(len(y) < 8 for y in outputs)

    if unpadded_count > padded_count:
        return "unpadded"

    return "padded"


def solve_bit_prompt_expr(prompt: str):
    examples = extract_bit_examples(prompt)
    query = extract_bit_query(prompt)

    if not examples or not query:
        return "", None, 0.0

    rule = learn_rule(examples)
    pred = predict(rule, query)

    # fmt = infer_bit_output_format(examples)

    # if fmt == "unpadded":
    #     pred = pred.lstrip("0") or "0"

    train_acc = rule_train_accuracy(rule, examples)
    rule_names = [item["name"] for item in rule]

    return pred, rule_names, train_acc


def main():
    df = pd.read_csv("data/train.csv")
    bit_df = df[df["prompt"].str.contains("8-bit binary", case=False)].copy()

    rows = []

    for _, row in bit_df.iterrows():
        pred, rule_names, train_acc = solve_bit_prompt_expr(row["prompt"])
        rows.append({
            "id": row["id"],
            "pred": pred,
            "answer": row["answer"],
            "correct": str(pred) == str(row["answer"]),
            "train_acc": train_acc,
            "rule": "|".join(rule_names) if rule_names else "",
        })

    out = pd.DataFrame(rows)
    out.to_csv("outputs/bit_expr_search_results.csv", index=False)

    print("bit tasks:", len(out))
    print("bit expr accuracy:", out["correct"].mean())
    print("\nTrain accuracy distribution:")
    print(out["train_acc"].value_counts().sort_index())
    print("\nsaved outputs/bit_expr_search_results.csv")


if __name__ == "__main__":
    main()
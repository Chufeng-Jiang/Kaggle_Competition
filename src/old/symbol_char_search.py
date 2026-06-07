import re
import pandas as pd

from old.parser import detect_task_type


def extract_pairs(prompt):
    pairs = []
    for line in prompt.splitlines():
        line = line.strip()
        if " = " in line and not line.startswith("In "):
            x, y = line.split(" = ", 1)
            pairs.append((x.strip(), y.strip()))
    return pairs


def extract_query(prompt):
    m = re.search(r"result for: (.*)", prompt)
    return m.group(1).strip() if m else ""


def is_char_expr(x):
    x = str(x)
    if len(x) != 5:
        return False
    return not (x[:2].isdigit() and x[3:].isdigit())


def char_templates(x):
    x = str(x)
    if len(x) != 5:
        return {}

    chars = list(x)
    left = x[:2]
    op = x[2]
    right = x[3:]
    full = left + right

    c = {}

    for mask in range(1, 32):
        idxs = [i for i in range(5) if mask & (1 << i)]
        c["pos_" + "".join(map(str, idxs))] = "".join(chars[i] for i in idxs)

    c["left"] = left
    c["right"] = right
    c["left_right"] = left + right
    c["right_left"] = right + left
    c["remove_op"] = x.replace(op, "")
    c["rev_full"] = full[::-1]

    c["firsts"] = left[0] + right[0]
    c["seconds"] = left[1] + right[1]
    c["outer"] = left[0] + right[1]
    c["inner"] = left[1] + right[0]

    c["unique_full"] = "".join(dict.fromkeys(full))
    c["unique_expr"] = "".join(dict.fromkeys(x))

    c["common_left"] = "".join(ch for ch in left if ch in right)
    c["common_right"] = "".join(ch for ch in right if ch in left)

    c["diff_left"] = "".join(ch for ch in left if ch not in right)
    c["diff_right"] = "".join(ch for ch in right if ch not in left)

    c["symdiff"] = "".join(ch for ch in full if full.count(ch) == 1)

    for i in range(5):
        for k in [2, 3, 4]:
            c[f"repeat_{i}_{k}"] = chars[i] * k

    return c


def solve_char_template(prompt):
    pairs = extract_pairs(prompt)
    query = extract_query(prompt)

    if not is_char_expr(query):
        return "", None, 0

    query_op = query[2]

    char_pairs = [(x, y) for x, y in pairs if is_char_expr(x)]
    same_op = [(x, y) for x, y in char_pairs if x[2] == query_op]

    used = same_op if same_op else char_pairs

    scores = {}

    for x, y in used:
        for rule, val in char_templates(x).items():
            if val == y:
                scores[rule] = scores.get(rule, 0) + 1

    if not scores:
        return "", None, 0

    best = max(scores.values())
    possible = [r for r, s in scores.items() if s == best]

    chosen = sorted(possible)[0]
    pred = char_templates(query).get(chosen, "")

    return pred, chosen, best


def main():
    df = pd.read_csv("data/train.csv")
    rows = []

    for _, row in df.iterrows():
        prompt = row["prompt"]

        if detect_task_type(prompt) != "symbol":
            continue

        query = extract_query(prompt)

        if not is_char_expr(query):
            continue

        pred, rule, score = solve_char_template(prompt)

        rows.append({
            "id": row["id"],
            "query_x": query,
            "pred": pred,
            "answer": str(row["answer"]),
            "correct": pred == str(row["answer"]),
            "rule": rule,
            "score": score,
        })

    out = pd.DataFrame(rows)
    out.to_csv("outputs/symbol_char_search_results.csv", index=False)

    print("char tasks:", len(out))
    print("accuracy:", out["correct"].mean())
    print("solved:", (out["pred"].astype(str) != "").sum())

    print("\nTop rules:")
    print(out["rule"].value_counts(dropna=False).head(30))

    print("\nsaved outputs/symbol_char_search_results.csv")


if __name__ == "__main__":
    main()
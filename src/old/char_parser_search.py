import re
import itertools
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


def remove_chars(s, delete_set):
    return "".join(ch for ch in s if ch not in delete_set)


def keep_chars(s, keep_set):
    return "".join(ch for ch in s if ch in keep_set)


def reverse_after(s, fn, charset):
    return fn(s, charset)[::-1]


def swap_halves_after_delete(s, delete_set):
    t = remove_chars(s, delete_set)
    mid = len(t) // 2
    return t[mid:] + t[:mid]


def generate_charsets(examples, max_size=3):
    chars = sorted(set("".join(x + y for x, y in examples)))

    sets = []

    for k in range(1, max_size + 1):
        for comb in itertools.combinations(chars, k):
            sets.append(set(comb))

    return sets


def split_by_control(s, ctrl):
    if ctrl not in s:
        return None

    parts = s.split(ctrl, 1)

    if len(parts) != 2:
        return None

    return parts[0], parts[1]


def candidate_programs(examples):
    programs = []

    # 只重点枚举高置信 control symbols
    control_symbols = ["+", "*"]

    for ctrl in control_symbols:
        # delete control
        programs.append((
            f"delete_{ctrl}",
            lambda x, ctrl=ctrl: remove_chars(x, {ctrl})
        ))

        # delete + reverse
        programs.append((
            f"delete_rev_{ctrl}",
            lambda x, ctrl=ctrl: remove_chars(x, {ctrl})[::-1]
        ))

        # delete + swap halves
        programs.append((
            f"delete_swap_{ctrl}",
            lambda x, ctrl=ctrl: swap_halves_after_delete(x, {ctrl})
        ))

        # delete + rotate
        for k in [1, 2, 3]:
            programs.append((
                f"delete_rotl{k}_{ctrl}",
                lambda x, ctrl=ctrl, k=k:
                    remove_chars(x, {ctrl})[k:] + remove_chars(x, {ctrl})[:k]
                    if len(remove_chars(x, {ctrl})) > k else ""
            ))

            programs.append((
                f"delete_rotr{k}_{ctrl}",
                lambda x, ctrl=ctrl, k=k:
                    remove_chars(x, {ctrl})[-k:] + remove_chars(x, {ctrl})[:-k]
                    if len(remove_chars(x, {ctrl})) > k else ""
            ))
            
        # delete control + position mask on cleaned string
        # cleaned length is usually 4 after removing + or *
        for mask in range(1, 1 << 5):
            def make_posmask_fn(ctrl, mask):
                def fn(x):
                    cleaned = remove_chars(x, {ctrl})
                    if not cleaned:
                        return ""

                    out = []
                    for i in range(len(cleaned)):
                        if mask & (1 << i):
                            out.append(cleaned[i])

                    return "".join(out)

                return fn

            name = f"delete_posmask_{mask}_{ctrl}"
            programs.append((
                name,
                make_posmask_fn(ctrl, mask)
            ))

        # delete control + reversed position mask
        for mask in range(1, 1 << 5):
            def make_rev_posmask_fn(ctrl, mask):
                def fn(x):
                    cleaned = remove_chars(x, {ctrl})[::-1]
                    if not cleaned:
                        return ""

                    out = []
                    for i in range(len(cleaned)):
                        if mask & (1 << i):
                            out.append(cleaned[i])

                    return "".join(out)

                return fn

            name = f"delete_rev_posmask_{mask}_{ctrl}"
            programs.append((
                name,
                make_rev_posmask_fn(ctrl, mask)
            ))

        # Treat ctrl as separator: A ctrl B
        programs.append((
            f"sep_lr_{ctrl}",
            lambda x, ctrl=ctrl:
                "".join(split_by_control(x, ctrl))
                if split_by_control(x, ctrl) else ""
        ))

        programs.append((
            f"sep_rl_{ctrl}",
            lambda x, ctrl=ctrl:
                split_by_control(x, ctrl)[1] + split_by_control(x, ctrl)[0]
                if split_by_control(x, ctrl) else ""
        ))

        programs.append((
            f"sep_l_rev_r_{ctrl}",
            lambda x, ctrl=ctrl:
                split_by_control(x, ctrl)[0] + split_by_control(x, ctrl)[1][::-1]
                if split_by_control(x, ctrl) else ""
        ))

        programs.append((
            f"sep_rev_l_r_{ctrl}",
            lambda x, ctrl=ctrl:
                split_by_control(x, ctrl)[0][::-1] + split_by_control(x, ctrl)[1]
                if split_by_control(x, ctrl) else ""
        ))

        programs.append((
            f"sep_rev_both_{ctrl}",
            lambda x, ctrl=ctrl:
                split_by_control(x, ctrl)[0][::-1] + split_by_control(x, ctrl)[1][::-1]
                if split_by_control(x, ctrl) else ""
        ))

        programs.append((
            f"sep_r_rev_l_{ctrl}",
            lambda x, ctrl=ctrl:
                split_by_control(x, ctrl)[1] + split_by_control(x, ctrl)[0][::-1]
                if split_by_control(x, ctrl) else ""
        ))

        programs.append((
            f"sep_rev_r_l_{ctrl}",
            lambda x, ctrl=ctrl:
                split_by_control(x, ctrl)[1][::-1] + split_by_control(x, ctrl)[0]
                if split_by_control(x, ctrl) else ""
        ))

        # keep only left / right segment
        programs.append((
            f"sep_left_{ctrl}",
            lambda x, ctrl=ctrl:
                split_by_control(x, ctrl)[0]
                if split_by_control(x, ctrl) else ""
        ))

        programs.append((
            f"sep_right_{ctrl}",
            lambda x, ctrl=ctrl:
                split_by_control(x, ctrl)[1]
                if split_by_control(x, ctrl) else ""
        ))

        programs.append((
            f"sep_left_rev_{ctrl}",
            lambda x, ctrl=ctrl:
                split_by_control(x, ctrl)[0][::-1]
                if split_by_control(x, ctrl) else ""
        ))

        programs.append((
            f"sep_right_rev_{ctrl}",
            lambda x, ctrl=ctrl:
                split_by_control(x, ctrl)[1][::-1]
                if split_by_control(x, ctrl) else ""
        ))

    # 保留原来的泛化 delete/keep，但限制 max_size=2，减少误伤
    charsets = generate_charsets(examples, max_size=2)

    for s in charsets:
        name = "delete_" + "".join(sorted(s))
        programs.append((
            name,
            lambda x, s=s: remove_chars(x, s)
        ))

        name = "keep_" + "".join(sorted(s))
        programs.append((
            name,
            lambda x, s=s: keep_chars(x, s)
        ))

        name = "delete_rev_" + "".join(sorted(s))
        programs.append((
            name,
            lambda x, s=s: remove_chars(x, s)[::-1]
        ))

        name = "keep_rev_" + "".join(sorted(s))
        programs.append((
            name,
            lambda x, s=s: keep_chars(x, s)[::-1]
        ))

        name = "delete_swap_" + "".join(sorted(s))
        programs.append((
            name,
            lambda x, s=s: swap_halves_after_delete(x, s)
        ))

    return programs


def solve_char_parser(prompt):
    pairs = extract_pairs(prompt)
    query = extract_query(prompt)

    if not is_char_expr(query):
        return "", None, 0

    query_op = query[2]

    char_pairs = [
        (x, y)
        for x, y in pairs
        if is_char_expr(x)
    ]

    same_op = [
        (x, y)
        for x, y in char_pairs
        if x[2] == query_op
    ]

    # 注意：parser 类规则通常不一定依赖 query op
    # 所以如果 same_op 太少，就用全部 char examples
    if len(same_op) >= 2:
        used = same_op
    else:
        used = char_pairs

    if not used:
        return "", None, 0

    programs = candidate_programs(used)

    candidates = []

    for name, fn in programs:
        score = 0

        for x, y in used:
            try:
                pred = fn(x)
            except Exception:
                pred = ""

            if pred == y:
                score += 1

        if score > 0:
            try:
                q_pred = fn(query)
            except Exception:
                q_pred = ""

            if q_pred:
                candidates.append((score, -len(name), name, q_pred))

    if not candidates:
        return "", None, 0

    candidates.sort(reverse=True)
    score, _, name, pred = candidates[0]

    return pred, name, score


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

        pred, rule, score = solve_char_parser(prompt)

        rows.append({
            "id": row["id"],
            "query": query,
            "pred": pred,
            "answer": str(row["answer"]),
            "correct": pred == str(row["answer"]),
            "rule": rule,
            "score": score,
        })

    out = pd.DataFrame(rows)
    out.to_csv("outputs/char_parser_results.csv", index=False)

    print("char tasks:", len(out))
    print("accuracy:", out["correct"].mean())
    print("solved:", (out["pred"].astype(str) != "").sum())

    print("\nTop rules:")
    print(out["rule"].value_counts(dropna=False).head(30))

    print("\nCorrect examples:")
    print(out[out["correct"] == True].head(30).to_string(index=False))

    print("\nWrong examples:")
    print(out[out["correct"] == False].head(30).to_string(index=False))


if __name__ == "__main__":
    main()
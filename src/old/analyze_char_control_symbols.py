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


def remove_char(s, ch):
    return "".join(c for c in s if c != ch)


def rotate_left(s, k):
    if not s:
        return s
    k %= len(s)
    return s[k:] + s[:k]


def rotate_right(s, k):
    if not s:
        return s
    k %= len(s)
    return s[-k:] + s[:-k]


def main():
    df = pd.read_csv("data/train.csv")

    input_count = Counter()
    output_count = Counter()

    appears_in_input_not_output = Counter()
    appears_in_output_not_input = Counter()

    delete_hits = Counter()
    delete_rev_hits = Counter()
    delete_rot_hits = Counter()
    delete_swap_hits = Counter()

    total_pairs_with_char = Counter()

    escape_cases = []

    total_char_pairs = 0

    for _, row in df.iterrows():
        prompt = row["prompt"]

        if detect_task_type(prompt) != "symbol":
            continue

        pairs = extract_pairs(prompt)

        for x, y in pairs:
            if not is_char_expr(x):
                continue

            total_char_pairs += 1

            x_chars = set(x)
            y_chars = set(y)

            for ch in x:
                input_count[ch] += 1

            for ch in y:
                output_count[ch] += 1

            for ch in x_chars:
                total_pairs_with_char[ch] += 1

                if ch not in y_chars:
                    appears_in_input_not_output[ch] += 1

                removed = remove_char(x, ch)

                if removed == y:
                    delete_hits[ch] += 1

                if removed[::-1] == y:
                    delete_rev_hits[ch] += 1

                # delete + rotate
                for k in range(1, max(1, len(removed))):
                    if rotate_left(removed, k) == y or rotate_right(removed, k) == y:
                        delete_rot_hits[ch] += 1
                        break

                # delete + swap halves
                if removed:
                    mid = len(removed) // 2
                    swapped = removed[mid:] + removed[:mid]
                    if swapped == y:
                        delete_swap_hits[ch] += 1

            for ch in y_chars:
                if ch not in x_chars:
                    appears_in_output_not_input[ch] += 1

            # escape pattern: backslash followed by char
            if "\\" in x:
                for i in range(len(x) - 1):
                    if x[i] == "\\":
                        escape_cases.append({
                            "x": x,
                            "y": y,
                            "escaped_char": x[i + 1],
                            "escaped_in_output": x[i + 1] in y,
                            "backslash_in_output": "\\" in y,
                        })

    rows = []

    all_chars = sorted(set(input_count) | set(output_count))

    for ch in all_chars:
        inp = input_count[ch]
        out = output_count[ch]
        pair_cnt = total_pairs_with_char[ch]

        rows.append({
            "char": ch,
            "input_count": inp,
            "output_count": out,
            "io_ratio_output_over_input": out / inp if inp else None,

            "pairs_with_char": pair_cnt,

            "input_not_output": appears_in_input_not_output[ch],
            "input_not_output_rate": (
                appears_in_input_not_output[ch] / pair_cnt
                if pair_cnt else 0
            ),

            "output_not_input": appears_in_output_not_input[ch],

            "delete_hits": delete_hits[ch],
            "delete_hit_rate": delete_hits[ch] / pair_cnt if pair_cnt else 0,

            "delete_rev_hits": delete_rev_hits[ch],
            "delete_rot_hits": delete_rot_hits[ch],
            "delete_swap_hits": delete_swap_hits[ch],

            "parser_hits_total": (
                delete_hits[ch]
                + delete_rev_hits[ch]
                + delete_rot_hits[ch]
                + delete_swap_hits[ch]
            ),
        })

    out = pd.DataFrame(rows)

    out = out.sort_values(
        ["parser_hits_total", "delete_hit_rate", "input_not_output_rate"],
        ascending=False,
    )

    out.to_csv("outputs/char_control_symbol_stats.csv", index=False)

    print("total char pairs:", total_char_pairs)

    print("\nTop possible control symbols:")
    print(
        out[
            [
                "char",
                "input_count",
                "output_count",
                "pairs_with_char",
                "input_not_output_rate",
                "delete_hits",
                "delete_hit_rate",
                "delete_rev_hits",
                "delete_rot_hits",
                "delete_swap_hits",
                "parser_hits_total",
            ]
        ].head(30).to_string(index=False)
    )

    print("\nTop generated symbols:")
    gen = out.sort_values("output_not_input", ascending=False)
    print(
        gen[
            [
                "char",
                "input_count",
                "output_count",
                "output_not_input",
                "io_ratio_output_over_input",
            ]
        ].head(30).to_string(index=False)
    )

    if escape_cases:
        esc = pd.DataFrame(escape_cases)
        esc.to_csv("outputs/char_escape_cases.csv", index=False)

        print("\nEscape cases:")
        print("num escape cases:", len(esc))
        print("escaped char in output rate:", esc["escaped_in_output"].mean())
        print("backslash in output rate:", esc["backslash_in_output"].mean())
        print(esc.head(30).to_string(index=False))

    print("\nsaved outputs/char_control_symbol_stats.csv")
    print("saved outputs/char_escape_cases.csv if escape cases exist")


if __name__ == "__main__":
    main()
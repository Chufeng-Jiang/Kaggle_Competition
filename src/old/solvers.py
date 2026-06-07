"""solvers.py"""

import re
import numpy as np

# ------------------

def format_number(x: float) -> str:
    x = round(x + 1e-12, 2)

    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))

    return f"{x:.2f}"


def fit_multiplier_by_rounding(xs, ys):
    """
    Find k such that round(k * x, 2) == y for all examples.
    If no exact interval exists, fall back to least squares.
    """
    low = -1e18
    high = 1e18

    for x, y in zip(xs, ys):
        x = float(x)
        y = float(y)

        # round(k*x, 2) = y roughly means:
        # y - 0.005 <= k*x < y + 0.005
        lo = (y - 0.005) / x
        hi = (y + 0.005) / x

        low = max(low, lo)
        high = min(high, hi)

    if low <= high:
        return (low + high) / 2

    xs = np.array(xs, dtype=float)
    ys = np.array(ys, dtype=float)

    return float(np.sum(xs * ys) / np.sum(xs * xs))


# -----------------

def solve_roman(prompt: str) -> str:
    m = re.search(r"write the number (\d+)", prompt.lower())
    if not m:
        return ""

    n = int(m.group(1))

    table = [
        (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
        (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
        (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")
    ]

    ans = ""
    for val, sym in table:
        while n >= val:
            ans += sym
            n -= val

    return ans


def solve_gravity(prompt: str) -> str:
    examples = re.findall(
        r"For t = ([\d.]+)s, distance = ([\d.]+) m",
        prompt
    )

    target = re.search(
        r"for t = ([\d.]+)s",
        prompt.split("Now")[-1]
    )

    if not examples or not target:
        return ""

    xs = []
    ys = []

    for t, d in examples:
        t = float(t)
        d = float(d)
        xs.append(0.5 * t * t)
        ys.append(d)

    low = -1e18
    high = 1e18

    for x, y in zip(xs, ys):
        lo = (y - 0.005) / x
        hi = (y + 0.005) / x
        low = max(low, lo)
        high = min(high, hi)

    if low <= high:
        alpha = 0.515
        g = low + alpha * (high - low)
    else:
        xs = np.array(xs)
        ys = np.array(ys)
        g = float(np.sum(xs * ys) / np.sum(xs * xs))

    t_new = float(target.group(1))
    pred = g * (0.5 * t_new * t_new)

    return format_number(pred)

def solve_unit(prompt: str) -> str:
    examples = re.findall(
        r"([\d.]+) m becomes ([\d.]+)",
        prompt
    )

    target = re.search(
        r"following measurement: ([\d.]+) m",
        prompt
    )

    if not examples or not target:
        return ""

    xs = []
    ys = []

    for x, y in examples:
        xs.append(float(x))
        ys.append(float(y))

    k = fit_multiplier_by_rounding(xs, ys)

    x_new = float(target.group(1))
    pred = k * x_new

    return format_number(pred)



def solve_cipher(prompt: str, vocab: set) -> str:
    # 读取 examples: encrypted -> plaintext
    pairs = re.findall(
        r"^([a-z ]+) -> ([a-z ]+)$",
        prompt,
        flags=re.M
    )

    enc_to_dec = {}
    dec_to_enc = {}

    for enc, dec in pairs:
        enc_chars = enc.replace(" ", "")
        dec_chars = dec.replace(" ", "")

        if len(enc_chars) != len(dec_chars):
            continue

        for e, d in zip(enc_chars, dec_chars):
            enc_to_dec[e] = d
            dec_to_enc[d] = e

    # 读取要 decrypt 的句子
    m = re.search(
        r"decrypt the following text: ([a-z ]+)",
        prompt.lower()
    )

    if not m:
        return ""

    target = m.group(1).strip()
    result_words = []

    for enc_word in target.split():
        decoded = "".join(
            enc_to_dec.get(ch, "?")
            for ch in enc_word
        )

        # 如果所有字母都能直接解出来
        if "?" not in decoded:
            result_words.append(decoded)
            continue

        # 如果有未知字母，用训练集答案词表补全
        candidates = []

        for word in vocab:
            if len(word) != len(enc_word):
                continue

            ok = True

            for e_ch, d_ch in zip(enc_word, word):
                # 已知映射必须一致
                if e_ch in enc_to_dec and enc_to_dec[e_ch] != d_ch:
                    ok = False
                    break

                # 反向映射也要一致，保证 substitution cipher
                if d_ch in dec_to_enc and dec_to_enc[d_ch] != e_ch:
                    ok = False
                    break

            if ok:
                candidates.append(word)

        if len(candidates) >= 1:
            result_words.append(candidates[0])
        else:
            result_words.append(decoded)

    return " ".join(result_words)


# -----------------
# Numeric Symbol Solver
# -----------------

def parse_numeric_expr(x: str):
    x = str(x).strip()

    if len(x) != 5:
        return None

    left = x[:2]
    op = x[2]
    right = x[3:]

    if not (left.isdigit() and right.isdigit()):
        return None

    return {
        "x": x,
        "left": left,
        "op": op,
        "right": right,
        "a": int(left),
        "b": int(right),
        "ar": int(left[::-1]),
        "br": int(right[::-1]),
        "l0": int(left[0]),
        "l1": int(left[1]),
        "r0": int(right[0]),
        "r1": int(right[1]),
    }


def rev_num_string(s: str):
    s = str(s)

    if s.startswith("-"):
        return "-" + s[1:][::-1]

    return s[::-1]


def numeric_candidate_outputs(x: str):
    p = parse_numeric_expr(x)

    if p is None:
        return {}

    a, b = p["a"], p["b"]
    ar, br = p["ar"], p["br"]
    l0, l1, r0, r1 = p["l0"], p["l1"], p["r0"], p["r1"]
    op = p["op"]

    c = {}

    def add_string_variants(name, s):
        s = str(s)

        c[name] = s
        c[name + "__revout"] = rev_num_string(s)

        c[name + "__op_prefix"] = op + s
        c[name + "__op_suffix"] = s + op

        rev = rev_num_string(s)
        c[name + "__revout_op_prefix"] = op + rev
        c[name + "__revout_op_suffix"] = rev + op

    def add(name, value):
        add_string_variants(name, str(value))

    def add_with_pm1(name, value):
        add(name, value)
        add(name + "_plus1", value + 1)
        add(name + "_minus1", value - 1)

    # -------------------------
    # normal arithmetic
    # -------------------------
    add_with_pm1("a_plus_b", a + b)
    add_with_pm1("a_minus_b", a - b)
    add_with_pm1("b_minus_a", b - a)
    add_with_pm1("abs_a_minus_b", abs(a - b))
    add_with_pm1("a_mul_b", a * b)

    # -------------------------
    # reversed operands arithmetic
    # -------------------------
    add_with_pm1("ar_plus_br", ar + br)
    add_with_pm1("ar_minus_br", ar - br)
    add_with_pm1("br_minus_ar", br - ar)
    add_with_pm1("abs_ar_minus_br", abs(ar - br))
    add_with_pm1("ar_mul_br", ar * br)

    # -------------------------
    # digit sums / products
    # -------------------------
    add_with_pm1("sum_all_digits", l0 + l1 + r0 + r1)
    add_with_pm1("sum_left_digits", l0 + l1)
    add_with_pm1("sum_right_digits", r0 + r1)
    add_with_pm1("abs_sum_digits_diff", abs((l0 + l1) - (r0 + r1)))

    add_with_pm1("prod_left_digits", l0 * l1)
    add_with_pm1("prod_right_digits", r0 * r1)
    add_with_pm1("sum_digit_products_parallel", l0 * r0 + l1 * r1)
    add_with_pm1("sum_digit_products_cross", l0 * r1 + l1 * r0)

    # -------------------------
    # digit-wise string outputs
    # -------------------------
    digit_candidates = {
        "digit_add_parallel": str(l0 + r0) + str(l1 + r1),
        "digit_add_cross": str(l0 + r1) + str(l1 + r0),

        "digit_absdiff_parallel": str(abs(l0 - r0)) + str(abs(l1 - r1)),
        "digit_absdiff_cross": str(abs(l0 - r1)) + str(abs(l1 - r0)),

        "digit_mul_parallel": str(l0 * r0) + str(l1 * r1),
        "digit_mul_cross": str(l0 * r1) + str(l1 * r0),

        "digit_sum_then_absdiff": str(l0 + l1) + str(abs(r0 - r1)),
        "digit_absdiff_then_sum": str(abs(l0 - l1)) + str(r0 + r1),

        "digit_parallel_pairs": str(l0) + str(r0) + str(l1) + str(r1),
        "digit_cross_pairs": str(l0) + str(r1) + str(l1) + str(r0),

        "digit_reverse_parallel_pairs": str(r0) + str(l0) + str(r1) + str(l1),
        "digit_reverse_cross_pairs": str(r1) + str(l0) + str(r0) + str(l1),
    }

    for name, s in digit_candidates.items():
        add_string_variants(name, s)

    # -------------------------
    # concatenation
    # -------------------------
    concat_candidates = {
        "left_right": p["left"] + p["right"],
        "right_left": p["right"] + p["left"],

        "rev_left_right": p["left"][::-1] + p["right"],
        "left_rev_right": p["left"] + p["right"][::-1],
        "rev_left_rev_right": p["left"][::-1] + p["right"][::-1],

        "left_right_revout": (p["left"] + p["right"])[::-1],
        "right_left_revout": (p["right"] + p["left"])[::-1],
    }

    for name, s in concat_candidates.items():
        add_string_variants(name, s)

    return c


def score_numeric_rules(examples):
    rule_scores = {}

    for x, y in examples:
        cands = numeric_candidate_outputs(x)

        for name, value in cands.items():
            if str(value) == str(y):
                rule_scores[name] = rule_scores.get(name, 0) + 1

    return rule_scores


def find_best_numeric_rules(examples):
    scores = score_numeric_rules(examples)

    if not scores:
        return set()

    best_score = max(scores.values())

    return {
        rule
        for rule, score in scores.items()
        if score == best_score
    }


def choose_numeric_rule(possible):
    if not possible:
        return None

    # priority = [
    #     # absolute forms first, because many ambiguous tasks want positive difference
    #     "abs_a_minus_b",
    #     "abs_a_minus_b__op_prefix",
    #     "abs_a_minus_b__op_suffix",
    #     "abs_a_minus_b__revout",
    #     "abs_a_minus_b__revout_op_prefix",
    #     "abs_a_minus_b__revout_op_suffix",

    #     "abs_ar_minus_br",
    #     "abs_ar_minus_br__op_prefix",
    #     "abs_ar_minus_br__op_suffix",
    #     "abs_ar_minus_br__revout",
    #     "abs_ar_minus_br__revout_op_prefix",
    #     "abs_ar_minus_br__revout_op_suffix",

    #     "a_plus_b",
    #     "a_plus_b__revout",
    #     "a_plus_b__op_prefix",
    #     "a_plus_b__op_suffix",

    #     "ar_plus_br",
    #     "ar_plus_br__revout",
    #     "ar_plus_br__op_prefix",
    #     "ar_plus_br__op_suffix",

    #     "a_minus_b",
    #     "a_minus_b__revout",
    #     "a_minus_b__op_prefix",
    #     "a_minus_b__op_suffix",

    #     "b_minus_a",
    #     "b_minus_a__revout",
    #     "b_minus_a__op_prefix",
    #     "b_minus_a__op_suffix",

    #     "ar_minus_br",
    #     "ar_minus_br__revout",
    #     "ar_minus_br__op_prefix",
    #     "ar_minus_br__op_suffix",

    #     "br_minus_ar",
    #     "br_minus_ar__revout",
    #     "br_minus_ar__op_prefix",
    #     "br_minus_ar__op_suffix",

    #     "a_mul_b",
    #     "a_mul_b__revout",
    #     "a_mul_b__op_prefix",
    #     "a_mul_b__op_suffix",

    #     "ar_mul_br",
    #     "ar_mul_br__revout",
    #     "ar_mul_br__op_prefix",
    #     "ar_mul_br__op_suffix",

    #     "digit_add_parallel",
    #     "digit_add_cross",
    #     "digit_absdiff_parallel",
    #     "digit_absdiff_cross",
    #     "digit_mul_parallel",
    #     "digit_mul_cross",

    #     "left_right",
    #     "right_left",
    #     "rev_left_right",
    #     "left_rev_right",
    #     "rev_left_rev_right",
    # ]
    
    priority = [
        "ar_plus_br__revout",
        "ar_mul_br__revout",
        "left_right",
        "rev_left_rev_right__revout",
        "ar_mul_br_minus1__revout",
        "ar_plus_br_minus1__revout",
        "ar_mul_br_plus1__revout",
        "ar_plus_br_plus1__revout",
        "ar_minus_br__revout",
        "abs_a_minus_b",
        "a_plus_b_plus1",
        "a_plus_b_minus1",
        "a_mul_b_plus1",
        "a_mul_b_minus1",
        "a_mul_b",
        "a_plus_b",
        "abs_ar_minus_br__revout_op_prefix",
        "abs_ar_minus_br__revout",
        "a_minus_b",
        "abs_a_minus_b__revout_op_prefix",
        "abs_ar_minus_br__revout_op_suffix",
        "a_minus_b__op_prefix",
        "a_minus_b__op_suffix",
    ]

    for rule in priority:
        if rule in possible:
            return rule

    return sorted(possible)[0]


def solve_numeric_symbol_prompt(prompt: str) -> str:
    pairs = []

    for line in prompt.splitlines():
        line = line.strip()

        if " = " in line and not line.startswith("In "):
            x, y = line.split(" = ", 1)
            pairs.append((x.strip(), y.strip()))

    m = re.search(r"result for: (.*)", prompt)
    if not m:
        return ""

    query_x = m.group(1).strip()
    query_info = parse_numeric_expr(query_x)

    if query_info is None:
        return ""

    query_op = query_info["op"]

    # 优先只看 query operator 相同的 examples
    same_op_examples = [
        (x, y)
        for x, y in pairs
        if parse_numeric_expr(x) is not None
        and parse_numeric_expr(x)["op"] == query_op
    ]

    if len(same_op_examples) > 0:
        used_examples = same_op_examples
    else:
        used_examples = [
            (x, y)
            for x, y in pairs
            if parse_numeric_expr(x) is not None
        ]

    if not used_examples:
        return ""

    possible = find_best_numeric_rules(used_examples)
    rule = choose_numeric_rule(possible)

    if rule is None:
        return ""

    return numeric_candidate_outputs(query_x).get(rule, "")


# -----------------
# Simple Character Symbol Solver
# -----------------

def char_symbol_candidates(expr: str):
    expr = str(expr).strip()

    if len(expr) != 5:
        return {}

    left = expr[:2]
    op = expr[2]
    right = expr[3:]

    cands = {}

    cands["concat_lr"] = left + right
    cands["concat_rl"] = right + left
    cands["left"] = left
    cands["right"] = right

    cands["firsts"] = left[0] + right[0]
    cands["seconds"] = left[1] + right[1]
    cands["outer"] = left[0] + right[1]
    cands["inner"] = left[1] + right[0]

    cands["rev_concat_lr"] = (left + right)[::-1]
    cands["rev_left_right"] = left[::-1] + right
    cands["left_rev_right"] = left + right[::-1]
    cands["rev_left_rev_right"] = left[::-1] + right[::-1]

    cands["unique_lr"] = "".join(dict.fromkeys(left + right))
    cands["unique_rl"] = "".join(dict.fromkeys(right + left))

    cands["common_lr"] = "".join(ch for ch in left if ch in right)
    cands["common_rl"] = "".join(ch for ch in right if ch in left)

    cands["diff_lr"] = "".join(ch for ch in left if ch not in right)
    cands["diff_rl"] = "".join(ch for ch in right if ch not in left)

    cands["symdiff_lr"] = "".join(
        ch for ch in left + right
        if (left + right).count(ch) == 1
    )

    # fixed position masks
    chars = list(expr)
    for mask in range(1, 32):
        idxs = [i for i in range(5) if mask & (1 << i)]
        name = "pos_" + "".join(map(str, idxs))
        cands[name] = "".join(chars[i] for i in idxs)

    return cands


def solve_char_symbol_prompt(prompt: str) -> str:
    pairs = []

    for line in prompt.splitlines():
        line = line.strip()

        if " = " in line and not line.startswith("In "):
            x, y = line.split(" = ", 1)
            pairs.append((x.strip(), y.strip()))

    m = re.search(r"result for: (.*)", prompt)
    if not m:
        return ""

    target = m.group(1).strip()

    if len(target) != 5:
        return ""

    target_op = target[2]

    same_op_examples = [
        (expr, out)
        for expr, out in pairs
        if len(expr) == 5 and expr[2] == target_op
    ]

    if same_op_examples:
        examples = same_op_examples
    else:
        examples = [
            (expr, out)
            for expr, out in pairs
            if len(expr) == 5
        ]

    scores = {}

    for expr, out in examples:
        cands = char_symbol_candidates(expr)

        for name, value in cands.items():
            if value == out:
                scores[name] = scores.get(name, 0) + 1

    if not scores:
        return ""

    best_score = max(scores.values())
    possible = [
        name
        for name, score in scores.items()
        if score == best_score
    ]

    chosen = sorted(possible)[0]

    return char_symbol_candidates(target).get(chosen, "")


def solve_symbol(prompt: str) -> str:
    # 1. numeric symbol solver first
    numeric_pred = solve_numeric_symbol_prompt(prompt)
    if numeric_pred != "":
        return numeric_pred

    # 2. fallback to simple char solver
    return solve_char_symbol_prompt(prompt)



# ---------------------

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


def extract_bit_examples(prompt: str):
    return re.findall(r"([01]{8}) -> ([01]{8})", prompt)


def extract_bit_query(prompt: str):
    m = re.search(r"determine the output for: ([01]{8})", prompt)
    if not m:
        return ""
    return m.group(1)


def build_expr_library():
    exprs = []

    # constant 0 / 1
    exprs.append(("const_0", lambda x: 0))
    exprs.append(("const_1", lambda x: 1))

    # xi and not xi
    for i in range(8):
        exprs.append((f"x{i}", lambda x, i=i: x[i]))
        exprs.append((f"not_x{i}", lambda x, i=i: 1 - x[i]))

    # binary expressions
    for i in range(8):
        for j in range(8):
            exprs.append(
                (
                    f"x{i}_xor_x{j}",
                    lambda x, i=i, j=j: x[i] ^ x[j],
                )
            )

            exprs.append(
                (
                    f"x{i}_and_x{j}",
                    lambda x, i=i, j=j: x[i] & x[j],
                )
            )

            exprs.append(
                (
                    f"x{i}_or_x{j}",
                    lambda x, i=i, j=j: x[i] | x[j],
                )
            )

    # ternary expressions
    for i in range(8):
        for j in range(8):
            for k in range(8):
                exprs.append(
                    (
                        f"maj_{i}_{j}_{k}",
                        lambda x, i=i, j=j, k=k: majority(
                            x[i],
                            x[j],
                            x[k],
                        ),
                    )
                )

                exprs.append(
                    (
                        f"ch_{i}_{j}_{k}",
                        lambda x, i=i, j=j, k=k: choice(
                            x[i],
                            x[j],
                            x[k],
                        ),
                    )
                )

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
        name, fn, score = learn_one_output_bit(
            examples,
            out_idx,
        )

        rule.append({
            "out_idx": out_idx,
            "name": name,
            "fn": fn,
            "score": score,
        })

    return rule


def predict(rule, x_bits):
    x = bits_to_list(x_bits)
    out = []

    for item in rule:
        out.append(item["fn"](x))

    return list_to_bits(out)


def rule_train_accuracy(rule, examples):
    correct = 0

    for x_bits, y_bits in examples:
        pred = predict(rule, x_bits)

        if pred == y_bits:
            correct += 1

    return correct / len(examples)


def solve_bit_prompt_expr(prompt: str):
    examples = extract_bit_examples(prompt)
    query = extract_bit_query(prompt)

    if not examples or not query:
        return "", None, 0.0

    rule = learn_rule(examples)
    pred = predict(rule, query)
    train_acc = rule_train_accuracy(rule, examples)

    rule_names = [item["name"] for item in rule]

    return pred, rule_names, train_acc

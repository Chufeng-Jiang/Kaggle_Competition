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


def symbol_candidates(expr: str):
    if len(expr) != 5:
        return {}

    left = expr[:2]
    op = expr[2]
    right = expr[3:]

    cands = {}

    # Basic string operations
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

    # Order-preserving set operations
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

    # Numeric operations
    if left.isdigit() and right.isdigit():
        a = int(left)
        b = int(right)

        cands["add"] = str(a + b)
        cands["abs_sub"] = str(abs(a - b))
        cands["sub_lr"] = str(a - b)
        cands["sub_rl"] = str(b - a)
        cands["mul"] = str(a * b)

        # Digit-wise operations
        l0, l1 = int(left[0]), int(left[1])
        r0, r1 = int(right[0]), int(right[1])

        cands["digit_add"] = str(l0 + r0) + str(l1 + r1)
        cands["digit_absdiff"] = str(abs(l0 - r0)) + str(abs(l1 - r1))
        cands["digit_mul"] = str(l0 * r0) + str(l1 * r1)

    return cands


def solve_symbol(prompt: str) -> str:
    pairs = []

    for line in prompt.splitlines():
        if " = " in line and not line.startswith("In "):
            left, right = line.split(" = ", 1)
            pairs.append((left, right))

    m = re.search(r"result for: (.*)", prompt)
    if not m:
        return ""

    target = m.group(1)
    if len(target) != 5:
        return ""

    target_op = target[2]

    # Prefer examples with the same operator
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

    possible = None

    for expr, out in examples:
        cands = symbol_candidates(expr)

        matched = {
            name
            for name, value in cands.items()
            if value == out
        }

        if possible is None:
            possible = matched
        else:
            possible = possible & matched

    if not possible:
        return ""

    # Deterministic choice
    chosen = sorted(possible)[0]

    return symbol_candidates(target).get(chosen, "")
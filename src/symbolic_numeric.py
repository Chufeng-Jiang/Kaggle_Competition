"""
symbol_digit_solver.py
SYMBOL-DIGIT 暴力求解器
输入格式: AB⊕CD = output，直接扫784种组合（频率加权）
"""

import re
import sys
from collections import Counter

import pandas as pd

# ============================================================
# 784种组合定义
# ============================================================

SN_OPERAND_MODES = [
    ("BA_DC", lambda a_str, b_str: (int(a_str[::-1]), int(b_str[::-1]))),
    ("AB_CD", lambda a_str, b_str: (int(a_str), int(b_str))),
    ("AB_DC", lambda a_str, b_str: (int(a_str), int(b_str[::-1]))),
    ("BA_CD", lambda a_str, b_str: (int(a_str[::-1]), int(b_str))),
]

SN_OPERATIONS = [
    ("add",         lambda a, b: a + b),
    ("mul",         lambda a, b: a * b),
    ("sub",         lambda a, b: a - b),
    ("rsub",        lambda a, b: b - a),
    ("abs_diff",    lambda a, b: abs(a - b)),
    ("add1",        lambda a, b: a + b + 1),
    ("add_m1",      lambda a, b: a + b - 1),
    ("mul_add1",    lambda a, b: a * b + 1),
    ("mul_sub1",    lambda a, b: a * b - 1),
    ("max",         lambda a, b: max(a, b)),
    ("min",         lambda a, b: min(a, b)),
    ("max_mod_min", lambda a, b: max(a, b) % min(a, b) if min(a, b) != 0 else 0),
    ("cat",         lambda a, b: int(str(a) + str(b))),
    ("rcat",        lambda a, b: int(str(b) + str(a))),
]

SN_FORMATS = [
    ("raw",           lambda x, op: str(x)),
    ("rev",           lambda x, op: str(x)[::-1]),
    ("abs",           lambda x, op: str(abs(x))),
    ("neg",           lambda x, op: str(-x)),
    ("neg_rev",       lambda x, op: str(-x)[::-1]),
    ("zpad2",         lambda x, op: str(abs(x)).zfill(2)),
    ("zpad3",         lambda x, op: str(abs(x)).zfill(3)),
    ("zpad4",         lambda x, op: str(abs(x)).zfill(4)),
    ("zpad2_rev",     lambda x, op: str(abs(x)).zfill(2)[::-1]),
    ("zpad3_rev",     lambda x, op: str(abs(x)).zfill(3)[::-1]),
    ("zpad4_rev",     lambda x, op: str(abs(x)).zfill(4)[::-1]),
    ("op_prefix",     lambda x, op: op + str(x)),
    ("op_suffix",     lambda x, op: str(x) + op),
    ("rev_op_prefix", lambda x, op: op + str(x)[::-1]),
    ("rev_op_suffix", lambda x, op: str(x)[::-1] + op),
    ("abs_op_prefix", lambda x, op: op + str(abs(x))),
    ("abs_op_suffix", lambda x, op: str(abs(x)) + op),
    ("dsum",          lambda x, op: str(sum(int(d) for d in str(abs(x))))),
]

# 频率加权顺序（从训练数据统计得出）
FREQ_ORDERED = [
    ("BA_DC", "add",         "rev"),
    ("BA_DC", "add",         "zpad2_rev"),
    ("BA_DC", "mul",         "rev"),
    ("BA_DC", "mul",         "zpad2_rev"),
    ("BA_DC", "mul",         "zpad3_rev"),
    ("BA_DC", "mul",         "zpad4_rev"),
    ("BA_DC", "abs_diff",    "rev"),
    ("BA_DC", "cat",         "rev"),
    ("BA_DC", "cat",         "zpad2_rev"),
    ("BA_DC", "cat",         "zpad3_rev"),
    ("BA_DC", "cat",         "zpad4_rev"),
    ("AB_CD", "cat",         "raw"),
    ("AB_CD", "cat",         "abs"),
    ("AB_CD", "cat",         "zpad2"),
    ("AB_CD", "cat",         "zpad3"),
    ("AB_CD", "cat",         "zpad4"),
    ("AB_CD", "sub",         "abs"),
    ("AB_CD", "rsub",        "abs"),
    ("AB_CD", "abs_diff",    "raw"),
    ("AB_CD", "abs_diff",    "abs"),
    ("BA_DC", "rcat",        "zpad4_rev"),
    ("BA_DC", "abs_diff",    "rev_op_prefix"),
    ("BA_DC", "mul_sub1",    "rev"),
    ("BA_DC", "mul_sub1",    "zpad2_rev"),
    ("BA_DC", "mul_sub1",    "zpad3_rev"),
    ("BA_DC", "max_mod_min", "rev"),
    ("BA_DC", "sub",         "zpad2_rev"),
    ("BA_DC", "rsub",        "zpad2_rev"),
    ("BA_DC", "abs_diff",    "zpad2_rev"),
    ("AB_CD", "rcat",        "zpad4"),
    ("BA_DC", "add_m1",      "rev"),
    ("BA_DC", "add_m1",      "zpad2_rev"),
    ("AB_CD", "max_mod_min", "raw"),
    ("AB_CD", "max_mod_min", "abs"),
    ("BA_DC", "mul_add1",    "rev"),
    ("BA_DC", "mul_add1",    "zpad2_rev"),
    ("BA_DC", "mul_add1",    "zpad3_rev"),
    ("BA_DC", "add",         "zpad3_rev"),
    ("BA_DC", "rcat",        "rev"),
    ("BA_DC", "rcat",        "zpad2_rev"),
    ("BA_DC", "rcat",        "zpad3_rev"),
    ("AB_CD", "sub",         "zpad2"),
    ("AB_CD", "rsub",        "zpad2"),
    ("AB_CD", "abs_diff",    "zpad2"),
    ("AB_CD", "rcat",        "raw"),
    ("AB_CD", "rcat",        "abs"),
    ("AB_CD", "rcat",        "zpad2"),
    ("AB_CD", "rcat",        "zpad3"),
    ("AB_CD", "sub",         "raw"),
    ("AB_CD", "rsub",        "neg"),
    ("AB_CD", "sub",         "op_suffix"),
    ("AB_CD", "rsub",        "op_prefix"),
    ("AB_CD", "abs_diff",    "op_prefix"),
    ("AB_CD", "abs_diff",    "op_suffix"),
    ("AB_CD", "sub",         "abs_op_prefix"),
    ("AB_CD", "rsub",        "abs_op_prefix"),
    ("AB_CD", "abs_diff",    "abs_op_prefix"),
    ("AB_CD", "max_mod_min", "op_prefix"),
    ("AB_CD", "max_mod_min", "op_suffix"),
    ("BA_DC", "sub",         "rev_op_prefix"),
    ("BA_DC", "rsub",        "rev_op_prefix"),
    ("BA_DC", "abs_diff",    "rev_op_suffix"),
    ("AB_CD", "add1",        "raw"),
    ("AB_CD", "add1",        "abs"),
    ("AB_CD", "add1",        "zpad2"),
    ("AB_CD", "add1",        "zpad3"),
    ("AB_CD", "sub",         "neg"),
    ("AB_CD", "rsub",        "raw"),
]

MODE_MAP = {n: f for n, f in SN_OPERAND_MODES}
OP_MAP   = {n: f for n, f in SN_OPERATIONS}
FMT_MAP  = {n: f for n, f in SN_FORMATS}

# 完整扫描顺序：高频优先，剩余补全
_FREQ_SET = set(FREQ_ORDERED)
FULL_SCAN_ORDER = FREQ_ORDERED + [
    (mn, on, fn)
    for mn, _ in SN_OPERAND_MODES
    for on, _ in SN_OPERATIONS
    for fn, _ in SN_FORMATS
    if (mn, on, fn) not in _FREQ_SET
]


# ============================================================
# 核心函数
# ============================================================

def apply_combo(left_str, right_str, op_str, mode_fn, op_fn, fmt_fn):
    try:
        a, b = mode_fn(left_str, right_str)
        result = op_fn(a, b)
        return fmt_fn(result, op_str)
    except Exception:
        return None


def scan_freq_first(decoded_examples, op_str):
    """
    频率优先扫描784种组合。
    examples >= 2条时第一个匹配直接返回；
    只有1条时收集所有匹配，返回频率最高的。
    """
    candidates = []

    for mode_name, op_name, fmt_name in FULL_SCAN_ORDER:
        mode_fn = MODE_MAP.get(mode_name)
        op_fn   = OP_MAP.get(op_name)
        fmt_fn  = FMT_MAP.get(fmt_name)
        if mode_fn is None or op_fn is None or fmt_fn is None:
            continue

        if all(apply_combo(l, r, op_str, mode_fn, op_fn, fmt_fn) == e
               for l, r, e in decoded_examples):
            candidates.append((mode_name, op_name, fmt_name, mode_fn, op_fn, fmt_fn))
            if len(decoded_examples) >= 2:
                return candidates[0]

    return candidates[0] if candidates else None


# ============================================================
# Prompt 解析
# ============================================================

def parse_symbol_prompt(prompt: str):
    """
    解析 symbol_numeric prompt。
    返回 (same_decoded, other_decoded, query, op_str)
    same_decoded:  [(left, right, out), ...]
    other_decoded: [(left, right, out, op_str), ...]
    """
    pairs = []
    for line in prompt.splitlines():
        line = line.strip()
        if " = " in line and not line.startswith("In "):
            x, y = line.split(" = ", 1)
            pairs.append((x.strip(), y.strip()))

    m = re.search(r"result for:\s*(.*)", prompt)
    if not m:
        return [], [], None, None

    query = m.group(1).strip()
    if len(query) != 5:
        return [], [], None, None

    op_str    = query[2]
    same_op   = [(x, y) for x, y in pairs if len(x) == 5 and x[2] == op_str]
    other_op  = [(x, y) for x, y in pairs if len(x) == 5 and x[2] != op_str]
    same_decoded  = [(x[:2], x[3:], y) for x, y in same_op]
    other_decoded = [(x[:2], x[3:], y, x[2]) for x, y in other_op]
    return same_decoded, other_decoded, query, op_str


# ============================================================
# 主求解函数
# ============================================================

def solve_symbol_numeric(prompt: str) -> str:
    """返回预测答案字符串，失败返回 ''"""
    same_decoded, other_decoded, query, op_str = parse_symbol_prompt(prompt)
    if query is None:
        return ""

    query_left  = query[:2]
    query_right = query[3:]

    if not same_decoded and not other_decoded:
        return ""

    if same_decoded:
        result = scan_freq_first(same_decoded, op_str)

        # same只有1条时，用other_decoded做二次验证缩小候选
        if result and len(same_decoded) <= 1 and other_decoded:
            all_valid = []
            for mn, mode_fn2 in SN_OPERAND_MODES:
                for on, op_fn2 in SN_OPERATIONS:
                    for fn, fmt_fn2 in SN_FORMATS:
                        if all(apply_combo(l, r, op_str, mode_fn2, op_fn2, fmt_fn2) == e
                               for l, r, e in same_decoded):
                            all_valid.append((mn, on, fn, mode_fn2, op_fn2, fmt_fn2))
            if len(all_valid) > 1:
                narrowed = []
                for mn, on, fn, mfn, ofn, ffn in all_valid:
                    score = sum(
                        1 for l, r, o, other_op in other_decoded
                        if any(apply_combo(l, r, other_op, mfn, ofn, ffn2) == o
                               for _, ffn2 in SN_FORMATS)
                    )
                    if score == len(other_decoded):
                        narrowed.append((mn, on, fn, mfn, ofn, ffn))
                if narrowed:
                    result = narrowed[0]

        if result is None:
            return ""
    else:
        decoded_3 = [(l, r, o) for l, r, o, _ in other_decoded]
        result = scan_freq_first(decoded_3, op_str)
        if result is None:
            return ""

    mode_name, op_name, fmt_name, mode_fn, op_fn, fmt_fn = result
    return apply_combo(query_left, query_right, op_str, mode_fn, op_fn, fmt_fn) or ""


def solve_symbol_numeric_with_trace(prompt: str) -> tuple:
    """返回 (pred, trace_str)，trace包含完整推理过程"""
    same_decoded, other_decoded, query, op_str = parse_symbol_prompt(prompt)
    if query is None:
        return "", "Parse failed"

    decoded_examples = same_decoded if same_decoded else [(l, r, o) for l, r, o, _ in other_decoded]
    if not decoded_examples:
        return "", "No examples"

    query_left  = query[:2]
    query_right = query[3:]

    result = scan_freq_first(decoded_examples, op_str)
    if result is None:
        return "", f"No rule found. Examples: {decoded_examples}"

    mode_name, op_name, fmt_name, mode_fn, op_fn, fmt_fn = result
    pred = apply_combo(query_left, query_right, op_str, mode_fn, op_fn, fmt_fn) or ""

    op_desc = {
        "add": "a + b", "sub": "a - b", "rsub": "b - a",
        "mul": "a * b", "abs_diff": "|a - b|",
        "add1": "a + b + 1", "add_m1": "a + b - 1",
        "mul_add1": "a * b + 1", "mul_sub1": "a * b - 1",
        "max": "max(a,b)", "min": "min(a,b)",
        "max_mod_min": "max(a,b) % min(a,b)",
        "cat": "str(a)+str(b)", "rcat": "str(b)+str(a)",
    }
    mode_desc = {
        "BA_DC": "reverse both operands",
        "AB_CD": "use operands as-is",
        "AB_DC": "left as-is, right reversed",
        "BA_CD": "left reversed, right as-is",
    }

    trace  = f"Step 1: Parse examples (operator='{op_str}'):\n"
    for left, right, out in decoded_examples:
        trace += f"  {left} {op_str} {right} = {out}\n"

    trace += f"\nStep 2: Scan 784 combinations (frequency-weighted).\n"
    trace += f"  LOCK: mode={mode_name} ({mode_desc.get(mode_name,'')})\n"
    trace += f"        op={op_name} ({op_desc.get(op_name,op_name)})\n"
    trace += f"        fmt={fmt_name}\n"

    trace += f"\nStep 3: Verify rule against all examples:\n"
    for left, right, expected in decoded_examples:
        a_val, b_val = mode_fn(left, right)
        raw    = op_fn(a_val, b_val)
        got    = apply_combo(left, right, op_str, mode_fn, op_fn, fmt_fn)
        status = "PASS" if got == expected else "FAIL"
        trace += f"  {left} {op_str} {right}: a={a_val}, b={b_val}, {op_desc.get(op_name,op_name)}={raw}, fmt='{got}' (expected '{expected}') [{status}]\n"

    a, b = mode_fn(query_left, query_right)
    raw  = op_fn(a, b)
    trace += f"\nStep 4: Apply rule to query '{query}':\n"
    trace += f"  a={a}, b={b}\n"
    trace += f"  {op_desc.get(op_name, op_name)} = {raw}\n"
    trace += f"  format '{fmt_name}' -> {pred}\n"
    trace += f"\n\\boxed{{{pred}}}"

    return pred, trace


# ============================================================
# 测试
# ============================================================

if __name__ == "__main__":
    sys.path.insert(0, "src")
    from generate_cot import normalize_task_type

    df = pd.read_csv("data/train.csv", dtype=str, keep_default_na=False)
    df["task_type"] = df.apply(lambda row: normalize_task_type(row.to_dict()), axis=1)
    symbol_numeric = df[df["task_type"] == "symbol_numeric"]
    print(f"symbol_numeric total: {len(symbol_numeric)}")

    correct, total, failed = 0, 0, []
    for _, row in symbol_numeric.iterrows():
        pred = solve_symbol_numeric(row["prompt"])
        gt   = row["answer"]
        if pred == gt:
            correct += 1
        else:
            failed.append({"prompt": row["prompt"], "gt": gt, "pred": pred})
        total += 1

    print(f"\nAccuracy: {correct}/{total} ({correct/total:.1%})")
    print(f"\nFailed cases ({len(failed)}):")
    for f in failed[:5]:
        same_decoded, other_decoded, query, op_str = parse_symbol_prompt(f["prompt"])
        decoded = same_decoded if same_decoded else other_decoded
        print(f"  GT: {f['gt']!r:15s} PRED: {f['pred']!r:15s} op='{op_str}' examples={decoded[:2]}")

    print("\n" + "=" * 60)
    print("Sample trace:")
    for _, row in symbol_numeric.iterrows():
        pred, trace = solve_symbol_numeric_with_trace(row["prompt"])
        if pred == row["answer"]:
            print(trace)
            break

    # 统计加权频率
    print("\nTop 30 weighted combos:")
    combo_counter: Counter = Counter()
    for _, row in symbol_numeric.iterrows():
        gt = row["answer"]
        same_decoded, other_decoded, query, op_str = parse_symbol_prompt(row["prompt"])
        if not same_decoded or query is None:
            continue
        query_left, query_right = query[:2], query[3:]
        matching = [
            (mn, on, fn)
            for mn, mode_fn in SN_OPERAND_MODES
            for on, op_fn in SN_OPERATIONS
            for fn, fmt_fn in SN_FORMATS
            if all(apply_combo(l, r, op_str, mode_fn, op_fn, fmt_fn) == e for l, r, e in same_decoded)
            and apply_combo(query_left, query_right, op_str, mode_fn, op_fn, fmt_fn) == gt
        ]
        if not matching:
            continue
        w = 1.0 / len(matching)
        for combo in matching:
            combo_counter[combo] += w
    for combo, count in combo_counter.most_common(30):
        print(f"  {combo[0]}|{combo[1]}|{combo[2]}: {count:.2f}")
"""
symbol_digit_solver.py
SYMBOL-DIGIT 暴力求解器
输入格式: AB⊕CD = output，直接扫784种组合
"""

import re
import pandas as pd
import sys
import os

# ============================================================
# 784种组合定义
# ============================================================

OPERAND_MODES = [
    ("AB_CD", lambda a_str, b_str: (int(a_str), int(b_str))),
    ("BA_DC", lambda a_str, b_str: (int(a_str[::-1]), int(b_str[::-1]))),
    ("AB_DC", lambda a_str, b_str: (int(a_str), int(b_str[::-1]))),
    ("BA_CD", lambda a_str, b_str: (int(a_str[::-1]), int(b_str))),
]  # 4种

# 2. 调整扫描优先级，把最常见的放前面
OPERAND_MODES = [
    ("BA_DC", lambda a_str, b_str: (int(a_str[::-1]), int(b_str[::-1]))),
    ("AB_CD", lambda a_str, b_str: (int(a_str), int(b_str))),
    ("AB_DC", lambda a_str, b_str: (int(a_str), int(b_str[::-1]))),
    ("BA_CD", lambda a_str, b_str: (int(a_str[::-1]), int(b_str))),
]

OPERATIONS = [
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
    ("max_mod_min", lambda a, b: max(a,b) % min(a,b) if min(a,b) != 0 else 0),
    ("cat",         lambda a, b: int(str(a) + str(b))),
    ("rcat",        lambda a, b: int(str(b) + str(a))),
]


FORMATS_SYMBOL = [
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

# 总计 4 × 14 × 14 = 784 种

FORMATS_ORDER = [
    "rev", "raw", "abs", "zpad2_rev", "zpad4_rev", "zpad3_rev",
    "zpad2", "zpad3", "zpad4",
    "op_prefix", "op_suffix", "rev_op_prefix", "rev_op_suffix",
    "abs_op_prefix", "abs_op_suffix",
    "neg", "neg_rev", "dsum",
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


def scan_784(decoded_examples, op_str):
    """
    decoded_examples: [(left_str, right_str, expected_out), ...]
    扫描所有784种组合，返回第一个匹配所有examples的组合
    返回 (mode_name, op_name, fmt_name, mode_fn, op_fn, fmt_fn) 或 None
    """
    for mode_name, mode_fn in OPERAND_MODES:
        for op_name, op_fn in OPERATIONS:
            for fmt_name, fmt_fn in FORMATS_SYMBOL:
                all_match = True
                for left, right, expected in decoded_examples:
                    result = apply_combo(left, right, op_str, mode_fn, op_fn, fmt_fn)
                    if result != expected:
                        all_match = False
                        break
                if all_match:
                    return mode_name, op_name, fmt_name, mode_fn, op_fn, fmt_fn
    return None


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

MODE_MAP = {n: f for n, f in OPERAND_MODES}
OP_MAP   = {n: f for n, f in OPERATIONS}
FMT_MAP  = {n: f for n, f in FORMATS_SYMBOL}

def scan_freq_first(decoded_examples, op_str):
    tried = set()
    candidates = []  # 收集所有匹配的combo

    # 第一遍：频率排序
    for mode_name, op_name, fmt_name in FREQ_ORDERED:
        mode_fn = MODE_MAP.get(mode_name)
        op_fn   = OP_MAP.get(op_name)
        fmt_fn  = FMT_MAP.get(fmt_name)
        if not mode_fn or not op_fn or not fmt_fn:
            continue
        tried.add((mode_name, op_name, fmt_name))
        if all(apply_combo(l, r, op_str, mode_fn, op_fn, fmt_fn) == e
               for l, r, e in decoded_examples):
            candidates.append((mode_name, op_name, fmt_name, mode_fn, op_fn, fmt_fn))
            # examples >= 2条时，第一个匹配就够可靠，直接返回
            if len(decoded_examples) >= 2:
                return candidates[0]

    # examples只有1条时，继续收集所有匹配
    # 第二遍：剩余combo
    for mode_name, mode_fn in OPERAND_MODES:
        for op_name, op_fn in OPERATIONS:
            for fmt_name, fmt_fn in FORMATS_SYMBOL:
                if (mode_name, op_name, fmt_name) in tried:
                    continue
                if all(apply_combo(l, r, op_str, mode_fn, op_fn, fmt_fn) == e
                       for l, r, e in decoded_examples):
                    candidates.append((mode_name, op_name, fmt_name, mode_fn, op_fn, fmt_fn))

    # 返回第一个（频率最高的）
    return candidates[0] if candidates else None
# ============================================================
# Prompt 解析
# ============================================================

# 先用同操作符examples，不够时加其他操作符的
def parse_symbol_prompt(prompt: str):
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

    op_str = query[2]

    same_op = [(x, y) for x, y in pairs if len(x) == 5 and x[2] == op_str]
    other_op = [(x, y) for x, y in pairs if len(x) == 5 and x[2] != op_str]

    same_decoded = [(x[:2], x[3:], y) for x, y in same_op]
    other_decoded = [(x[:2], x[3:], y, x[2]) for x, y in other_op]

    return same_decoded, other_decoded, query, op_str


# ============================================================
# 主求解函数
# ============================================================

def solve_symbol_numeric(prompt: str) -> str:
    same_decoded, other_decoded, query, op_str = parse_symbol_prompt(prompt)

    if query is None:
        return ""

    query_left = query[:2]
    query_right = query[3:]

    if not same_decoded and not other_decoded:
        return ""

    # Step 1: 用same_decoded扫描
    if same_decoded:
        result = scan_freq_first(same_decoded, op_str)
        
        # Step 2: 如果same_decoded只有1条且有多个候选，用other_decoded做二次验证
        if result and len(same_decoded) <= 1 and other_decoded:
            # 找出所有匹配same_decoded的combo
            all_valid = []
            for mn, mode_fn2 in OPERAND_MODES:
                for on, op_fn2 in OPERATIONS:
                    for fn, fmt_fn2 in FORMATS_SYMBOL:
                        if all(apply_combo(l, r, op_str, mode_fn2, op_fn2, fmt_fn2) == e
                               for l, r, e in same_decoded):
                            all_valid.append((mn, on, fn, mode_fn2, op_fn2, fmt_fn2))
            
            if len(all_valid) > 1:
                # 用other_decoded缩小：mode+op必须也能解释other_decoded
                narrowed = []
                for mn, on, fn, mfn, ofn, ffn in all_valid:
                    score = 0
                    for l, r, o, other_op in other_decoded:
                        for _, ffn2 in FORMATS_SYMBOL:
                            if apply_combo(l, r, other_op, mfn, ofn, ffn2) == o:
                                score += 1
                                break
                    if score == len(other_decoded):
                        narrowed.append((mn, on, fn, mfn, ofn, ffn))
                if narrowed:
                    result = narrowed[0]

    else:
        # 没有same_decoded，用other_decoded（3元组）
        decoded_3 = [(l, r, o) for l, r, o, _ in other_decoded]
        result = scan_freq_first(decoded_3, op_str)

    if result is None:
        return ""

    mode_name, op_name, fmt_name, mode_fn, op_fn, fmt_fn = result
    pred = apply_combo(query_left, query_right, op_str, mode_fn, op_fn, fmt_fn)
    return pred or ""


def solve_symbol_numeric_with_trace(prompt: str) -> tuple:
    same_decoded, other_decoded, query, op_str = parse_symbol_prompt(prompt)

    if query is None:
        return "", "Parse failed"

    if same_decoded:
        decoded_examples = same_decoded
    elif other_decoded:
        decoded_examples = [(l, r, o) for l, r, o, _ in other_decoded]
    else:
        return "", "No examples"

    query_left = query[:2]
    query_right = query[3:]

    result = scan_freq_first(decoded_examples, op_str)

    if result is None:
        trace = "Scanned all combinations, no match found.\n"
        trace += f"Examples: {decoded_examples}\n"
        trace += f"Query: {query}"
        return "", trace

    mode_name, op_name, fmt_name, mode_fn, op_fn, fmt_fn = result
    pred = apply_combo(query_left, query_right, op_str, mode_fn, op_fn, fmt_fn)

    trace = f"Step 1: Parse examples (operator='{op_str}'):\n"
    for left, right, out in decoded_examples:
        trace += f"  {left} {op_str} {right} = {out}\n"
    trace += f"\nStep 2: Scan combinations (4 modes × 14 ops × 18 formats).\n"
    trace += f"  LOCK: mode={mode_name}, op={op_name}, fmt={fmt_name}\n"
    trace += f"\nStep 3: Verify rule against all examples:\n"
    for left, right, expected in decoded_examples:
        got = apply_combo(left, right, op_str, mode_fn, op_fn, fmt_fn)
        status = "PASS" if got == expected else "FAIL"
        trace += f"  {left} {op_str} {right} -> {got} (expected {expected}) [{status}]\n"
    trace += f"\nStep 4: Apply to query '{query}':\n"
    a, b = mode_fn(query_left, query_right)
    trace += f"  mode={mode_name}: a={a}, b={b}\n"
    trace += f"  op={op_name}: result={op_fn(a, b)}\n"
    trace += f"  fmt={fmt_name}: final={pred}\n"
    trace += f"\n\\boxed{{{pred}}}"

    return pred or "", trace

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

    # 测试前100条
    test_df = symbol_numeric.head(732)
    correct = 0
    total = 0
    failed = []

    for _, row in test_df.iterrows():
        pred = solve_symbol_numeric(row["prompt"])
        gt = row["answer"]
        is_correct = (pred == gt)
        if is_correct:
            correct += 1
        else:
            failed.append({
                "prompt": row["prompt"],
                "gt": gt,
                "pred": pred,
            })
        total += 1

    print(f"\nAccuracy: {correct}/{total} ({correct/total:.1%})")

    # 打印失败案例
    print(f"\nFailed cases ({len(failed)}):")
# 在测试里加详细分析
    for f in failed[:10]:
        print(f"GT: {f['gt']!r}")
        print(f"PRED: {f['pred']!r}")
        
        same_decoded, other_decoded, query, op_str = parse_symbol_prompt(f['prompt'])
        decoded = same_decoded if same_decoded else other_decoded
        print(f"op_str: {op_str!r}")
        print(f"decoded examples: {decoded}")

        # 看看GT对应哪种组合
        query_left = query[:2]
        query_right = query[3:]
        gt = f['gt']
        
        for mode_name, mode_fn in OPERAND_MODES:
            for op_name, op_fn in OPERATIONS:
                for fmt_name, fmt_fn in FORMATS_SYMBOL:
                    try:
                        pred = apply_combo(query_left, query_right, op_str, mode_fn, op_fn, fmt_fn)
                        if pred == gt:
                            print(f"  GT combo: {mode_name}|{op_name}|{fmt_name}")
                    except:
                        pass
        print()
    
    for f in failed[:3]:
        print("=" * 60)
        print(f"GT: {f['gt']!r}  PRED: {f['pred']!r}")
        print(f"FULL PROMPT:")
        print(f['prompt'])
        print()

    # 打印一条成功案例的完整trace
    print("=" * 60)
    print("Sample trace:")
    for _, row in test_df.iterrows():
        pred, trace = solve_symbol_numeric_with_trace(row["prompt"])
        if pred == row["answer"]:
            print(trace)
            break

from collections import Counter
combo_counter = Counter()

for _, row in symbol_numeric.iterrows():
    gt = row["answer"]
    same_decoded, other_decoded, query, op_str = parse_symbol_prompt(row["prompt"])
    if not same_decoded or query is None:
        continue
    
    query_left = query[:2]
    query_right = query[3:]
    
    # 找所有能解释examples AND 得到GT的combo
    matching = []
    for mode_name, mode_fn in OPERAND_MODES:
        for op_name, op_fn in OPERATIONS:
            for fmt_name, fmt_fn in FORMATS_SYMBOL:
                if not all(apply_combo(l, r, op_str, mode_fn, op_fn, fmt_fn) == e
                           for l, r, e in same_decoded):
                    continue
                pred = apply_combo(query_left, query_right, op_str, mode_fn, op_fn, fmt_fn)
                if pred == gt:
                    matching.append((mode_name, op_name, fmt_name))
    
    if not matching:
        continue
    
    # 按1/n权重计入，n是匹配数量
    # 唯一匹配权重=1，10个匹配各权重=0.1
    weight = 1.0 / len(matching)
    for combo in matching:
        combo_counter[combo] += weight

print("Top 50 weighted combos:")
for combo, count in combo_counter.most_common(50):
    print(f"  {combo[0]}|{combo[1]}|{combo[2]}: {count:.2f}")
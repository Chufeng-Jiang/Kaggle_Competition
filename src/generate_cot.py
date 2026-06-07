import json
import os
import re
import time
import traceback
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from collections import defaultdict

import numpy as np
import pandas as pd
from openai import OpenAI


# ============================================================
# 配置
# ============================================================

client = OpenAI(
    api_key="sk-724636c17e2e4e0d8e6da790242e433f",
    base_url="https://api.deepseek.com",
)

MODEL = "deepseek-v4-flash"
TEMPERATURE = 0.2
MAX_TOKENS = 4096

DATA_PATH = "data/train.csv"
OUTPUT_PATH = "outputs/cot_data.csv"
ALL_OUTPUT_PATH = "outputs/cot_data_all.csv"
CHECKPOINT_PATH = "outputs/checkpoint.json"
MAX_WORKERS = 50


# ============================================================
# 工具函数
# ============================================================

def format_number(x: float) -> str:
    x = round(x + 1e-12, 2)
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return f"{x:.2f}"


def fit_multiplier_by_rounding(xs, ys):
    low, high = -1e18, 1e18
    for x, y in zip(xs, ys):
        x, y = float(x), float(y)
        low = max(low, (y - 0.005) / x)
        high = min(high, (y + 0.005) / x)
    if low <= high:
        return (low + high) / 2
    xs = np.array(xs, dtype=float)
    ys = np.array(ys, dtype=float)
    return float(np.sum(xs * ys) / np.sum(xs * xs))


# ============================================================
# Roman Solver
# ============================================================

def solve_roman(prompt: str) -> tuple:
    m = re.search(r"write the number (\d+)", prompt.lower())
    if not m:
        return "", "Parse failed"
    n = int(m.group(1))
    original_n = n
    table = [
        (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
        (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
        (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")
    ]
    ans = ""
    steps = []
    for val, sym in table:
        while n >= val:
            ans += sym
            steps.append(f"  {n} >= {val}, append '{sym}', remaining = {n - val}")
            n -= val
    cot = f"Convert {original_n} to Roman numeral step by step:\n"
    cot += "\n".join(steps) + "\n"
    cot += f"Result: {ans}\n\\boxed{{{ans}}}"
    return ans, cot


# ============================================================
# Gravity Solver
# ============================================================

def solve_gravity(prompt: str) -> tuple:
    examples = re.findall(r"For t = ([\d.]+)s, distance = ([\d.]+) m", prompt)
    target = re.search(r"for t = ([\d.]+)s", prompt.split("Now")[-1])
    if not examples or not target:
        return "", "Parse failed"
    xs, ys = [], []
    for t, d in examples:
        xs.append(0.5 * float(t) ** 2)
        ys.append(float(d))
    low, high = -1e18, 1e18
    for x, y in zip(xs, ys):
        low = max(low, (y - 0.005) / x)
        high = min(high, (y + 0.005) / x)
    if low <= high:
        g = low + 0.515 * (high - low)
    else:
        xs_arr, ys_arr = np.array(xs), np.array(ys)
        g = float(np.sum(xs_arr * ys_arr) / np.sum(xs_arr * xs_arr))
    t_new = float(target.group(1))
    pred_val = g * 0.5 * t_new ** 2
    pred = format_number(pred_val)
    cot = "Step 1: Extract examples and compute rate = d / (0.5 * t^2) for each:\n"
    for (t, d), x, y in zip(examples, xs, ys):
        cot += f"  t={t}s, d={d}m -> rate = {y}/{x:.4f} = {y/x:.4f}\n"
    cot += f"\nStep 2: Inferred g = {g:.4f} (via interval reasoning)\n"
    cot += f"\nStep 3: Apply to query t={t_new}s:\n"
    cot += f"  d = {g:.4f} * 0.5 * {t_new}^2 = {g:.4f} * {0.5*t_new**2:.4f} = {pred_val:.4f}\n"
    cot += f"\nRounded: {pred}\n\\boxed{{{pred}}}"
    return pred, cot


# ============================================================
# Unit Solver
# ============================================================

def solve_unit(prompt: str) -> tuple:
    examples = re.findall(r"([\d.]+) m becomes ([\d.]+)", prompt)
    target = re.search(r"following measurement: ([\d.]+) m", prompt)
    if not examples or not target:
        return "", "Parse failed"
    xs = [float(x) for x, _ in examples]
    ys = [float(y) for _, y in examples]
    k = fit_multiplier_by_rounding(xs, ys)
    x_new = float(target.group(1))
    pred_val = k * x_new
    pred = format_number(pred_val)
    cot = "Step 1: Compute rate k = output / input for each example:\n"
    for x, y in zip(xs, ys):
        cot += f"  {y} / {x} = {y/x:.6f}\n"
    cot += f"\nStep 2: Inferred k = {k:.6f} (via interval reasoning)\n"
    cot += f"\nStep 3: Apply to query {x_new}:\n"
    cot += f"  {x_new} * {k:.6f} = {pred_val:.6f}\n"
    cot += f"\nRounded: {pred}\n\\boxed{{{pred}}}"
    return pred, cot


# ============================================================
# Cipher Solver
# ============================================================

def build_vocab(data_path: str) -> set:
    try:
        df = pd.read_csv(data_path, dtype=str, keep_default_na=False)
        vocab = set()
        for ans in df["answer"]:
            ans = str(ans).strip()
            if re.match(r'^[a-z ]+$', ans):
                for word in ans.split():
                    vocab.add(word.lower())
        return vocab
    except Exception:
        return set()


def solve_cipher(prompt: str, vocab: set) -> tuple:
    pairs = re.findall(r"^([a-z ]+) -> ([a-z ]+)$", prompt, flags=re.M)
    enc_to_dec, dec_to_enc = {}, {}
    for enc, dec in pairs:
        enc_chars = enc.replace(" ", "")
        dec_chars = dec.replace(" ", "")
        if len(enc_chars) != len(dec_chars):
            continue
        for e, d in zip(enc_chars, dec_chars):
            enc_to_dec[e] = d
            dec_to_enc[d] = e
    m = re.search(r"decrypt the following text: ([a-z ]+)", prompt.lower())
    if not m:
        return "", "Parse failed"
    target = m.group(1).strip()
    result_words = []
    cot = "Step 1: Build substitution mapping from examples:\n"
    for enc, dec in pairs:
        cot += f"  '{enc}' -> '{dec}'\n"
    cot += f"\nFull mapping: {enc_to_dec}\n"
    cot += f"\nStep 2: Decrypt target '{target}' character by character:\n"
    for enc_word in target.split():
        decoded_chars = []
        for ch in enc_word:
            mapped = enc_to_dec.get(ch, "?")
            decoded_chars.append(mapped)
            cot += f"  '{ch}' -> '{mapped}'\n"
        decoded = "".join(decoded_chars)
        if "?" not in decoded:
            result_words.append(decoded)
        else:
            candidates = [
                word for word in vocab
                if len(word) == len(enc_word) and all(
                    (e not in enc_to_dec or enc_to_dec[e] == d) and
                    (d not in dec_to_enc or dec_to_enc[d] == e)
                    for e, d in zip(enc_word, word)
                )
            ]
            chosen = candidates[0] if candidates else decoded
            result_words.append(chosen)
            if candidates:
                cot += f"  '{decoded}' has gaps, vocab match: '{chosen}'\n"
    pred = " ".join(result_words)
    cot += f"\nStep 3: Final decrypted phrase: '{pred}'\n\\boxed{{{pred}}}"
    return pred, cot


# ============================================================
# Bit Solver (DFS trace)
# ============================================================

OPS = {
    'AND':         lambda a, b: a & b,
    'OR':          lambda a, b: a | b,
    'XOR':         lambda a, b: a ^ b,
    'NAND':        lambda a, b: ~(a & b),
    'NOR':         lambda a, b: ~(a | b),
    'XNOR':        lambda a, b: ~(a ^ b),
    'NOT_A_AND_B': lambda a, b: (~a) & b,
    'A_AND_NOT_B': lambda a, b: a & (~b),
    'NOT_A_OR_B':  lambda a, b: (~a) | b,
    'A_OR_NOT_B':  lambda a, b: a | (~b),
}

BIT_TRANSFORMATIONS = [('rot', 0)]
for _k in range(1, 8):
    BIT_TRANSFORMATIONS.extend([('rot', _k), ('shl', _k), ('shr', _k)])


def get_used_vars(expr):
    used = []
    if '{A}' in expr: used.append('{A}')
    if '{B}' in expr: used.append('{B}')
    if '{C}' in expr: used.append('{C}')
    return used


def get_source_bit(in_bits, out_idx, trans):
    ttype, shift_val = trans
    if ttype == 'rot':
        return in_bits[(out_idx + shift_val) % 8]
    elif ttype == 'shl':
        src = out_idx + shift_val
        return in_bits[src] if 0 <= src < 8 else 0
    elif ttype == 'shr':
        src = out_idx - shift_val
        return in_bits[src] if 0 <= src < 8 else 0


def evaluate_bit(evaluator, trans_dict, bit_idx, in_arrays, out_arrays, num_examples):
    for ex in range(num_examples):
        in_bits = in_arrays[ex]
        expected = out_arrays[ex][bit_idx]
        av = get_source_bit(in_bits, bit_idx, trans_dict.get('{A}', ('rot', 0)))
        bv = get_source_bit(in_bits, bit_idx, trans_dict.get('{B}', ('rot', 0)))
        cv = get_source_bit(in_bits, bit_idx, trans_dict.get('{C}', ('rot', 0)))
        if evaluator(av, bv, cv, 1) != expected:
            return False
    return True


def generate_grammar_dynamically():
    mask = 255
    l0 = {
        0:          ("C0",  lambda a, b, c, m: 0),
        255:        ("C1",  lambda a, b, c, m: m),
        0b11110000: ("{A}", lambda a, b, c, m: a),
        0b11001100: ("{B}", lambda a, b, c, m: b),
        0b10101010: ("{C}", lambda a, b, c, m: c),
    }
    visited = set(l0.keys())
    levels = [l0]
    for tt, (expr, func) in l0.items():
        yield tt, expr, func
    for depth in range(1, 4):
        next_level = {}
        for v, (expr, func) in levels[-1].items():
            not_v = (~v) & mask
            if not_v not in visited:
                new_expr = f"NOT({expr})"
                new_func = lambda a, b, c, m, f=func: (~f(a, b, c, m)) & m
                visited.add(not_v)
                next_level[not_v] = (new_expr, new_func)
                yield not_v, new_expr, new_func
        for i in range(depth):
            j = depth - 1
            for v1, (expr1, func1) in levels[i].items():
                for v2, (expr2, func2) in levels[j].items():
                    for op_name, op_func in OPS.items():
                        if i == j and v1 > v2 and op_name in ['AND','OR','XOR','NAND','NOR','XNOR']:
                            continue
                        val = op_func(v1, v2) & mask
                        if val not in visited:
                            new_expr = f"{op_name}({expr1}, {expr2})"
                            new_func = lambda a,b,c,m,f1=func1,f2=func2,op=op_func: op(f1(a,b,c,m),f2(a,b,c,m))&m
                            visited.add(val)
                            next_level[val] = (new_expr, new_func)
                            yield val, new_expr, new_func
                        if i != j:
                            val2 = op_func(v2, v1) & mask
                            if val2 not in visited:
                                new_expr = f"{op_name}({expr2}, {expr1})"
                                new_func = lambda a,b,c,m,f1=func1,f2=func2,op=op_func: op(f2(a,b,c,m),f1(a,b,c,m))&m
                                visited.add(val2)
                                next_level[val2] = (new_expr, new_func)
                                yield val2, new_expr, new_func
        levels.append(next_level)


def format_hyp(expr, trans_dict):
    s = expr
    for k, v in trans_dict.items():
        s = s.replace(k, str(v))
    return s


def solve_dfs_trace_dynamic(in_arrays, out_arrays, num_examples):
    trace = []
    start_time = time.time()
    grammar_gen = generate_grammar_dynamically()
    for tt, expr, evaluator in grammar_gen:
        if time.time() - start_time > 5.0:
            trace.append("TIMEOUT")
            return trace, None
        used = get_used_vars(expr)
        combinations = []
        if len(used) == 0:
            combinations.append({})
        elif len(used) == 1:
            for t1 in BIT_TRANSFORMATIONS:
                combinations.append({used[0]: t1})
        elif len(used) == 2:
            for t1 in BIT_TRANSFORMATIONS:
                for t2 in BIT_TRANSFORMATIONS:
                    if t1 == t2: continue
                    combinations.append({used[0]: t1, used[1]: t2})
        elif len(used) == 3:
            for t1, t2, t3 in itertools.permutations(BIT_TRANSFORMATIONS, 3):
                combinations.append({used[0]: t1, used[1]: t2, used[2]: t3})
        for trans_dict in combinations:
            if time.time() - start_time > 5.0:
                trace.append("TIMEOUT")
                return trace, None
            if evaluate_bit(evaluator, trans_dict, 0, in_arrays, out_arrays, num_examples):
                hyp_str = format_hyp(expr, trans_dict)
                trace.append(f"B0: Testing {hyp_str} -> YES")
                valid_global = True
                for b in range(1, 8):
                    if evaluate_bit(evaluator, trans_dict, b, in_arrays, out_arrays, num_examples):
                        trace.append(f"B{b}: Testing {hyp_str} -> YES")
                    else:
                        trace.append(f"B{b}: Testing {hyp_str} -> NO. Contradiction, backtracking...")
                        valid_global = False
                        break
                if valid_global:
                    trace.append(f"GLOBAL MATCH FOUND: {hyp_str}")
                    def predictor(q_in, trans_dict=dict(trans_dict), ev=evaluator):
                        res = []
                        for b_idx in range(8):
                            av = get_source_bit(q_in, b_idx, trans_dict.get('{A}', ('rot', 0)))
                            bv = get_source_bit(q_in, b_idx, trans_dict.get('{B}', ('rot', 0)))
                            cv = get_source_bit(q_in, b_idx, trans_dict.get('{C}', ('rot', 0)))
                            res.append(str(ev(av, bv, cv, 1)))
                        return "".join(res)
                    return trace, predictor
    trace.append("NO MATCH FOUND")
    return trace, None


def solve_bit(prompt: str) -> tuple:
    ex_matches = re.findall(r'([01]{8})\s*->\s*([01]{8})', prompt)
    if not ex_matches:
        return "", "No examples found"
    num_examples = len(ex_matches)
    in_arrays  = [[int(ex_matches[ex][0][j]) for j in range(8)] for ex in range(num_examples)]
    out_arrays = [[int(ex_matches[ex][1][j]) for j in range(8)] for ex in range(num_examples)]
    query_match = re.search(
        r'(?:output for:|determine the output for:)\s*([01]{8})',
        prompt, re.IGNORECASE
    )
    if not query_match:
        return "", "No query found"
    query_in = [int(b) for b in query_match.group(1)]
    trace, predictor = solve_dfs_trace_dynamic(in_arrays, out_arrays, num_examples)
    if predictor:
        pred = predictor(query_in)
        cot = "\n".join(trace) + f"\n\n\\boxed{{{pred}}}"
        return pred, cot
    return "", "\n".join(trace)


# ============================================================
# Symbol Numeric Solver (784种组合，频率加权扫描)
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
    ("max_mod_min", lambda a, b: max(a,b) % min(a,b) if min(a,b) != 0 else 0),
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
    ("BA_DC", "max_mod_min", "rev"),
    ("BA_DC", "abs_diff",    "rev_op_prefix"),
    ("BA_DC", "add_m1",      "rev"),
    ("BA_DC", "add_m1",      "zpad2_rev"),
    ("BA_DC", "mul",         "zpad4_rev"),
    ("BA_DC", "abs_diff",    "rev"),
    ("BA_DC", "add1",        "rev"),
    ("BA_DC", "add1",        "zpad2_rev"),
    ("BA_DC", "mul_sub1",    "rev"),
    ("BA_DC", "mul_sub1",    "zpad2_rev"),
    ("BA_DC", "mul_sub1",    "zpad3_rev"),
    ("BA_DC", "add",         "zpad3_rev"),
    ("BA_DC", "cat",         "rev"),
    ("BA_DC", "cat",         "zpad2_rev"),
    ("BA_DC", "cat",         "zpad3_rev"),
    ("BA_DC", "cat",         "zpad4_rev"),
    ("AB_CD", "add",         "zpad2"),
    ("BA_DC", "mul_add1",    "rev"),
    ("BA_DC", "mul_add1",    "zpad2_rev"),
    ("BA_DC", "mul_add1",    "zpad3_rev"),
    ("AB_CD", "add",         "raw"),
    ("AB_CD", "add",         "abs"),
    ("AB_CD", "max_mod_min", "raw"),
    ("AB_CD", "max_mod_min", "abs"),
    ("AB_CD", "cat",         "raw"),
    ("AB_CD", "cat",         "abs"),
    ("AB_CD", "cat",         "zpad2"),
    ("AB_CD", "cat",         "zpad3"),
    ("AB_CD", "add_m1",      "raw"),
    ("AB_CD", "add_m1",      "abs"),
    ("AB_CD", "add_m1",      "zpad2"),
    ("AB_CD", "cat",         "zpad4"),
    ("BA_DC", "rsub",        "rev_op_prefix"),
    ("AB_CD", "add1",        "raw"),
    ("AB_CD", "add1",        "abs"),
    ("AB_CD", "add1",        "zpad2"),
    ("BA_DC", "mul_add1",    "zpad4_rev"),
    ("BA_DC", "mul_sub1",    "zpad4_rev"),
    ("AB_CD", "sub",         "abs"),
    ("AB_CD", "rsub",        "abs"),
    ("AB_CD", "abs_diff",    "raw"),
    ("AB_CD", "abs_diff",    "abs"),
    ("BA_DC", "rcat",        "zpad4_rev"),
    ("AB_CD", "mul",         "raw"),
    ("AB_CD", "mul",         "abs"),
    ("BA_DC", "abs_diff",    "rev_op_suffix"),
    ("BA_DC", "sub",         "rev_op_suffix"),
    ("BA_DC", "rsub",        "rev_op_suffix"),
    ("AB_CD", "sub",         "op_suffix"),
    ("AB_CD", "abs_diff",    "op_suffix"),
    ("AB_CD", "rsub",        "op_prefix"),
    ("AB_CD", "abs_diff",    "op_prefix"),
    ("AB_CD", "sub",         "abs_op_prefix"),
    ("AB_CD", "rsub",        "abs_op_prefix"),
    ("AB_CD", "abs_diff",    "abs_op_prefix"),
    ("AB_CD", "max_mod_min", "op_prefix"),
    ("AB_CD", "max_mod_min", "op_suffix"),
]

MODE_MAP = {n: f for n, f in SN_OPERAND_MODES}
OP_MAP   = {n: f for n, f in SN_OPERATIONS}
FMT_MAP  = {n: f for n, f in SN_FORMATS}


def sn_apply_combo(left_str, right_str, op_str, mode_fn, op_fn, fmt_fn):
    try:
        a, b = mode_fn(left_str, right_str)
        result = op_fn(a, b)
        return fmt_fn(result, op_str)
    except Exception:
        return None


def scan_freq_first(decoded_examples, op_str):
    """频率优先扫描，examples>=2条时第一个匹配直接返回"""
    tried = set()

    for mode_name, op_name, fmt_name in FREQ_ORDERED:
        mode_fn = MODE_MAP.get(mode_name)
        op_fn   = OP_MAP.get(op_name)
        fmt_fn  = FMT_MAP.get(fmt_name)
        if not mode_fn or not op_fn or not fmt_fn:
            continue
        tried.add((mode_name, op_name, fmt_name))
        if all(sn_apply_combo(l, r, op_str, mode_fn, op_fn, fmt_fn) == e
               for l, r, e in decoded_examples):
            if len(decoded_examples) >= 2:
                return mode_name, op_name, fmt_name, mode_fn, op_fn, fmt_fn

    # examples只有1条时继续收集所有匹配，返回第一个高频的
    for mode_name, mode_fn in SN_OPERAND_MODES:
        for op_name, op_fn in SN_OPERATIONS:
            for fmt_name, fmt_fn in SN_FORMATS:
                if (mode_name, op_name, fmt_name) in tried:
                    continue
                if all(sn_apply_combo(l, r, op_str, mode_fn, op_fn, fmt_fn) == e
                       for l, r, e in decoded_examples):
                    return mode_name, op_name, fmt_name, mode_fn, op_fn, fmt_fn

    return None


def parse_symbol_numeric_prompt(prompt: str):
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


def solve_symbol_numeric(prompt: str) -> tuple:
    same_decoded, other_decoded, query, op_str = parse_symbol_numeric_prompt(prompt)
    if query is None:
        return "", "Parse failed"
    query_left = query[:2]
    query_right = query[3:]

    if same_decoded:
        decoded_examples = same_decoded
    elif other_decoded:
        decoded_examples = [(l, r, o) for l, r, o, _ in other_decoded]
    else:
        return "", "No examples"

    result = scan_freq_first(decoded_examples, op_str)
    if result is None:
        return "", "No rule found"

    mode_name, op_name, fmt_name, mode_fn, op_fn, fmt_fn = result
    pred = sn_apply_combo(query_left, query_right, op_str, mode_fn, op_fn, fmt_fn)
    if not pred:
        return "", "Apply failed"

    cot = f"Step 1: Parse examples (operator='{op_str}'):\n"
    for left, right, out in decoded_examples:
        cot += f"  {left} {op_str} {right} = {out}\n"
    cot += f"\nStep 2: Scan 784 combinations (frequency-weighted).\n"
    cot += f"  LOCK: mode={mode_name}, op={op_name}, fmt={fmt_name}\n"
    cot += f"\nStep 3: Verify rule against all examples:\n"
    for left, right, expected in decoded_examples:
        got = sn_apply_combo(left, right, op_str, mode_fn, op_fn, fmt_fn)
        status = "PASS" if got == expected else "FAIL"
        cot += f"  {left} {op_str} {right} -> {got} (expected {expected}) [{status}]\n"
    a, b = mode_fn(query_left, query_right)
    cot += f"\nStep 4: Apply to query '{query}':\n"
    cot += f"  mode={mode_name}: a={a}, b={b}\n"
    cot += f"  op={op_name}: result={op_fn(a, b)}\n"
    cot += f"  fmt={fmt_name}: final={pred}\n"
    cot += f"\n\\boxed{{{pred}}}"
    return pred, cot


# ============================================================
# Symbol Char Solver (旧版位置枚举，cipher_digit待Picat实现)
# ============================================================

def char_symbol_candidates(expr):
    expr = str(expr).strip()
    if len(expr) != 5: return {}
    left, op, right = expr[:2], expr[2], expr[3:]
    cands = {
        "concat_lr": left+right, "concat_rl": right+left,
        "left": left, "right": right,
        "firsts": left[0]+right[0], "seconds": left[1]+right[1],
        "outer": left[0]+right[1], "inner": left[1]+right[0],
        "rev_concat_lr": (left+right)[::-1],
        "rev_left_right": left[::-1]+right,
        "left_rev_right": left+right[::-1],
        "rev_left_rev_right": left[::-1]+right[::-1],
        "unique_lr": "".join(dict.fromkeys(left+right)),
        "unique_rl": "".join(dict.fromkeys(right+left)),
        "common_lr": "".join(ch for ch in left if ch in right),
        "common_rl": "".join(ch for ch in right if ch in left),
        "diff_lr": "".join(ch for ch in left if ch not in right),
        "diff_rl": "".join(ch for ch in right if ch not in left),
        "symdiff_lr": "".join(ch for ch in left+right if (left+right).count(ch)==1),
    }
    chars = list(expr)
    for mask in range(1, 32):
        idxs = [i for i in range(5) if mask & (1 << i)]
        cands["pos_"+"".join(map(str,idxs))] = "".join(chars[i] for i in idxs)
    return cands


def solve_symbol_char(prompt: str) -> tuple:
    pairs = []
    for line in prompt.splitlines():
        line = line.strip()
        if " = " in line and not line.startswith("In "):
            x, y = line.split(" = ", 1)
            pairs.append((x.strip(), y.strip()))
    m = re.search(r"result for: (.*)", prompt)
    if not m:
        return "", "Parse failed"
    target = m.group(1).strip()
    if len(target) != 5:
        return "", "Query length != 5"
    target_op = target[2]
    same_op = [(e,o) for e,o in pairs if len(e)==5 and e[2]==target_op]
    examples = same_op if same_op else [(e,o) for e,o in pairs if len(e)==5]
    scores = {}
    for expr, out in examples:
        for name, value in char_symbol_candidates(expr).items():
            if value == out:
                scores[name] = scores.get(name, 0) + 1
    if not scores:
        return "", "No rule found"
    best = max(scores.values())
    possible = [n for n, s in scores.items() if s == best]
    pred = char_symbol_candidates(target).get(sorted(possible)[0], "")
    if not pred:
        return "", "Apply failed"
    cot = f"Step 1: Parse examples (operator='{target_op}'):\n"
    for expr, out in examples:
        cot += f"  {expr} = {out}\n"
    cot += f"\nStep 2: Score candidate rules, best={sorted(possible)[0]}\n"
    cot += f"\nStep 3: Apply to query '{target}': {pred}\n"
    cot += f"\n\\boxed{{{pred}}}"
    return pred, cot


# ============================================================
# 题型识别
# ============================================================

def detect_task_type(prompt: str) -> str:
    p = str(prompt).lower()
    if "8-bit binary" in p or "bit manipulation" in p: return "bit"
    if "secret encryption" in p or "decrypt" in p: return "cipher"
    if "unit conversion" in p: return "unit"
    if "gravitational constant" in p: return "gravity"
    if "wonderland numeral system" in p: return "roman"
    if "transformation rules is applied to equations" in p: return "symbol"
    return "unknown"


def extract_symbol_query(prompt: str) -> str:
    m = re.search(r"result\s+for\s*:\s*(.+)", str(prompt), re.I | re.S)
    return m.group(1).strip() if m else ""


def is_numeric_symbol_query(query: str) -> bool:
    return re.match(r'^\d+([^\d]+\d+)+$', str(query).strip()) is not None


def normalize_task_type(row: dict) -> str:
    prompt = str(row.get("prompt", ""))
    raw_type = str(row.get("type", "")).lower()
    if "numeral" in raw_type or "roman" in raw_type: return "roman"
    if "encrypt" in raw_type or "cipher" in raw_type: return "cipher"
    if "unit" in raw_type: return "unit"
    if "gravity" in raw_type or "gravitational" in raw_type: return "gravity"
    if "bit" in raw_type: return "bit"
    base = "symbol" if ("equation" in raw_type or "symbol" in raw_type) else detect_task_type(prompt)
    if base == "symbol":
        query = extract_symbol_query(prompt)
        return "symbol_numeric" if is_numeric_symbol_query(query) else "symbol_char"
    return base


# ============================================================
# Prompt（LLM兜底用）
# ============================================================

PROMPTS = {
    "symbol_numeric": r"""
You are solving a numeric symbol transformation task.
Each expression: A...B?C...D -> output. Operator is context-specific.
Rules: a+b, a-b, b-a, |a-b|, a*b, rev(a)+rev(b), rev(result), concatenation, digit-wise ops.
Output may have operator prefix/suffix (e.g. @75, 61%).
Find a rule consistent with all examples sharing the same operator as the query.
Put ONLY the final answer string inside \boxed{}.
""",
    "symbol_char": r"""
You are solving a cipher-digit task. Every symbol is an encrypted digit (0-9).
1. DETECT operator (fixed position, usually index 2).
2. CRACK cipher: build symbol->digit mapping from examples.
3. SCAN arithmetic rule: BA+DC, BA*DC reversed, AB+CD, etc.
4. VERIFY against all examples.
5. APPLY to query and re-encode to cipher symbols.
Show symbol->digit mapping and state the rule.
Put ONLY the cipher-encoded answer inside \boxed{}.
""",
    "unknown": r"""Infer the rule. Verify. Apply. Put answer inside \boxed{}.""",
}


def build_user_message(prompt: str) -> str:
    return f"Here is the task prompt:\n\n{prompt}\n\nPut the final answer inside exactly one \\boxed{{...}}."


# ============================================================
# 答案提取和验证
# ============================================================

def extract_answer(response: str) -> str:
    response = str(response)
    start = response.rfind(r"\boxed{")
    if start != -1:
        i = start + len(r"\boxed{")
        depth, ans = 1, []
        while i < len(response):
            ch = response[i]
            if ch == "{": depth += 1; ans.append(ch)
            elif ch == "}":
                depth -= 1
                if depth == 0: break
                ans.append(ch)
            else: ans.append(ch)
            i += 1
        return "".join(ans).strip()
    m = re.search(r'"final_answer"\s*:\s*"([^"]*)"', response)
    if m: return m.group(1).strip()
    lines = [l.strip() for l in response.strip().splitlines() if l.strip()]
    return lines[-1] if lines else ""


def clean_boxed_answer(ans: str) -> str:
    ans = str(ans).strip()
    m = re.fullmatch(r"\\text\{(.+)\}", ans)
    return m.group(1).strip() if m else ans


def check_answer(pred: str, gt: str, task_type: str) -> bool:
    pred, gt = str(pred).strip(), str(gt).strip()
    if task_type == "bit": return pred == gt
    if task_type in {"gravity", "unit"}:
        try:
            pm = re.search(r"-?\d+(?:\.\d+)?", pred)
            gm = re.search(r"-?\d+(?:\.\d+)?", gt)
            if not pm or not gm: return False
            p, g = float(pm.group()), float(gm.group())
            return abs(p-g) <= 0.05 or abs(p-g)/max(abs(g),1e-9) <= 0.005
        except: return False
    if task_type == "roman":
        m = re.search(r"[IVXLCDM]+", pred.upper())
        return (m.group(0) if m else pred.upper()) == gt.upper()
    if task_type == "cipher": return pred.lower().strip() == gt.lower().strip()
    return pred == gt


# ============================================================
# API 调用
# ============================================================

def call_api(system_msg: str, user_msg: str) -> str:
    for attempt in range(4):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role":"system","content":system_msg},{"role":"user","content":user_msg}],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                extra_body={"thinking": {"type": "disabled"}},
            )
            content = resp.choices[0].message.content or ""
            if not content:
                print(f"[WARN] Empty response, finish_reason={resp.choices[0].finish_reason}")
            return content
        except Exception as e:
            print(f"API error attempt {attempt+1}: {e}")
            time.sleep(3*(attempt+1))
    print("[ERROR] All retries failed, returning empty")
    return ""


# ============================================================
# process_row
# ============================================================

VOCAB: set = set()


def process_row(row: dict) -> dict:
    prompt = str(row["prompt"])
    gt = str(row["answer"])
    task_type = normalize_task_type(row)

    if task_type == "bit":
        pred, cot = solve_bit(prompt)
    elif task_type == "gravity":
        pred, cot = solve_gravity(prompt)
    elif task_type == "unit":
        pred, cot = solve_unit(prompt)
    elif task_type == "roman":
        pred, cot = solve_roman(prompt)
    elif task_type == "cipher":
        pred, cot = solve_cipher(prompt, VOCAB)
    elif task_type == "symbol_numeric":
        pred, cot = solve_symbol_numeric(prompt)
        if not pred:
            system_msg = PROMPTS["symbol_numeric"]
            response = call_api(system_msg, build_user_message(prompt))
            pred = clean_boxed_answer(extract_answer(response))
            cot = response
    elif task_type == "symbol_char":
        pred, cot = solve_symbol_char(prompt)
        if not pred:
            system_msg = PROMPTS["symbol_char"]
            response = call_api(system_msg, build_user_message(prompt))
            pred = clean_boxed_answer(extract_answer(response))
            cot = response
    else:
        system_msg = PROMPTS.get(task_type, PROMPTS["unknown"])
        response = call_api(system_msg, build_user_message(prompt))
        pred = clean_boxed_answer(extract_answer(response))
        cot = response

    correct = check_answer(pred, gt, task_type)
    return {
        "id": row["id"], "type": task_type,
        "prompt": prompt, "answer": gt,
        "generated_cot": cot, "extracted_answer": pred, "correct": correct,
    }


# ============================================================
# 主流程
# ============================================================

def save_checkpoint(results):
    tmp_path = CHECKPOINT_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, CHECKPOINT_PATH)


def main():
    global VOCAB
    VOCAB = build_vocab(DATA_PATH)
    print(f"Vocab size: {len(VOCAB)}")

    df = pd.read_csv(DATA_PATH, dtype=str, keep_default_na=False)
    print(f"Total rows: {len(df)}")

    results = []
    done_ids = set()

    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
            results = json.load(f)
        done_ids = {str(r["id"]) for r in results}
        print(f"Loaded checkpoint: {len(done_ids)} finished")

    remaining = df[~df["id"].astype(str).isin(done_ids)].to_dict("records")
    remaining = [
        row for row in remaining
        if normalize_task_type(row) in {
            "bit", "gravity", "unit", "roman", "cipher",
            "symbol_numeric", "symbol_char",
        }
    ]
    print(f"Remaining rows: {len(remaining)}")

    lock = Lock()
    type_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in results:
        t = r["type"]
        type_stats[t]["total"] += 1
        if r.get("correct"):
            type_stats[t]["correct"] += 1

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_row, row): row for row in remaining}
        for i, future in enumerate(as_completed(futures), start=1):
            try:
                result = future.result()
            except Exception:
                row = futures[future]
                result = {
                    "id": row.get("id",""), "type": "error",
                    "prompt": row.get("prompt",""), "answer": row.get("answer",""),
                    "generated_cot": traceback.format_exc(),
                    "extracted_answer": "", "correct": False,
                }
            with lock:
                results.append(result)
                t = result["type"]
                type_stats[t]["total"] += 1
                if result["correct"]:
                    type_stats[t]["correct"] += 1
            if i % 50 == 0:
                save_checkpoint(results)
                print(f"\nProgress: {len(results)}/{len(df)}")
                for t, s in sorted(type_stats.items()):
                    acc = s["correct"]/s["total"] if s["total"] else 0
                    print(f"  {t}: {s['correct']}/{s['total']} ({acc:.1%})")

    save_checkpoint(results)

    result_df = pd.DataFrame(results)
    result_df.to_csv(ALL_OUTPUT_PATH, index=False, encoding="utf-8")

    correct_df = result_df[result_df["correct"] == True].copy()
    correct_df[["id","type","prompt","answer","generated_cot","extracted_answer"]].to_csv(
        OUTPUT_PATH, index=False, encoding="utf-8"
    )

    print()
    print(f"Saved all rows: {ALL_OUTPUT_PATH}")
    print(f"Saved correct rows: {OUTPUT_PATH}")
    print(f"Correct samples: {len(correct_df)}/{len(result_df)}")
    print("\nFinal stats:")
    for t, s in sorted(type_stats.items()):
        acc = s["correct"]/s["total"] if s["total"] else 0
        print(f"  {t}: {s['correct']}/{s['total']} ({acc:.1%})")


if __name__ == "__main__":
    main()
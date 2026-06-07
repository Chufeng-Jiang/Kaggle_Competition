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
# 从 solvers.py 迁移的工具函数
# ============================================================

def format_number(x: float) -> str:
    x = round(x + 1e-12, 2)
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return f"{x:.2f}"


def fit_multiplier_by_rounding(xs, ys):
    low = -1e18
    high = 1e18
    for x, y in zip(xs, ys):
        x, y = float(x), float(y)
        lo = (y - 0.005) / x
        hi = (y + 0.005) / x
        low = max(low, lo)
        high = min(high, hi)
    if low <= high:
        return (low + high) / 2
    xs = np.array(xs, dtype=float)
    ys = np.array(ys, dtype=float)
    return float(np.sum(xs * ys) / np.sum(xs * xs))


# ============================================================
# Roman Solver（来自 solvers.py）
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
# Gravity Solver（来自 solvers.py）
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
# Unit Solver（来自 solvers.py）
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
# Cipher Solver（来自 solvers.py）
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
# Bit Solver（来自 solvers.py）
# ============================================================

# def bits_to_list(s):
#     return [int(c) for c in s.strip()]

# def list_to_bits(xs):
#     return "".join(str(int(x)) for x in xs)

# def majority(a, b, c):
#     return int(a + b + c >= 2)

# def choice(a, b, c):
#     return b if a else c

# def build_expr_library():
#     exprs = []
#     exprs.append(("const_0", lambda x: 0))
#     exprs.append(("const_1", lambda x: 1))
#     for i in range(8):
#         exprs.append((f"x{i}", lambda x, i=i: x[i]))
#         exprs.append((f"not_x{i}", lambda x, i=i: 1 - x[i]))
#     for i in range(8):
#         for j in range(8):
#             exprs.append((f"x{i}_xor_x{j}", lambda x, i=i, j=j: x[i] ^ x[j]))
#             exprs.append((f"x{i}_and_x{j}", lambda x, i=i, j=j: x[i] & x[j]))
#             exprs.append((f"x{i}_or_x{j}",  lambda x, i=i, j=j: x[i] | x[j]))
#     for i in range(8):
#         for j in range(8):
#             for k in range(8):
#                 exprs.append((f"maj_{i}_{j}_{k}", lambda x, i=i, j=j, k=k: majority(x[i], x[j], x[k])))
#                 exprs.append((f"ch_{i}_{j}_{k}",  lambda x, i=i, j=j, k=k: choice(x[i], x[j], x[k])))
#     return exprs

# EXPR_LIBRARY = build_expr_library()

# def expr_complexity(name):
#     if name.startswith("const"):    return 5
#     if name.startswith("not_x"):   return 2
#     if re.fullmatch(r"x\d", name): return 1
#     if "_xor_" in name:            return 3
#     if "_and_" in name:            return 4
#     if "_or_" in name:             return 4
#     if name.startswith("maj_"):    return 5
#     if name.startswith("ch_"):     return 6
#     return 99

# def learn_one_output_bit(examples, output_idx):
#     best_name, best_fn, best_score, best_comp = None, None, -1, 999
#     for name, fn in EXPR_LIBRARY:
#         score = sum(1 for x_bits, y_bits in examples if fn(bits_to_list(x_bits)) == int(y_bits[output_idx]))
#         comp = expr_complexity(name)
#         if (score > best_score
#                 or (score == best_score and comp < best_comp)
#                 or (score == best_score and comp == best_comp and str(name) < str(best_name))):
#             best_name, best_fn, best_score, best_comp = name, fn, score, comp
#     return best_name, best_fn, best_score

# def learn_rule(examples):
#     return [{"out_idx": i, **dict(zip(["name","fn","score"], learn_one_output_bit(examples, i)))} for i in range(8)]

# def predict_bit(rule, x_bits):
#     x = bits_to_list(x_bits)
#     return list_to_bits([item["fn"](x) for item in rule])

# def rule_train_accuracy(rule, examples):
#     return sum(1 for x_bits, y_bits in examples if predict_bit(rule, x_bits) == y_bits) / len(examples)

# def solve_bit(prompt: str) -> tuple:
#     examples = re.findall(r"([01]{8}) -> ([01]{8})", prompt)
#     m = re.search(r"determine the output for: ([01]{8})", prompt)
#     if not examples or not m:
#         return "", "Parse failed"

#     query = m.group(1)
#     rule = learn_rule(examples)
#     pred = predict_bit(rule, query)
#     train_acc = rule_train_accuracy(rule, examples)

#     cot = "Step 1: Learn a boolean function for each output bit:\n"
#     for item in rule:
#         cot += f"  out[{item['out_idx']}] = {item['name']} (train score: {item['score']}/{len(examples)})\n"
#     cot += f"\nStep 2: Verify on all training examples:\n"
#     for x_bits, y_bits in examples:
#         p = predict_bit(rule, x_bits)
#         status = "PASS" if p == y_bits else "FAIL"
#         cot += f"  {x_bits} -> {p} (expected {y_bits}) [{status}]\n"
#     cot += f"\nTrain accuracy: {train_acc:.2f}\n"
#     cot += f"\nStep 3: Apply to query {query}:\n"
#     x = bits_to_list(query)
#     for item in rule:
#         bit_val = item["fn"](x)
#         cot += f"  out[{item['out_idx']}] = {item['name']}({query}) = {bit_val}\n"
#     cot += f"\nResult: {pred}\n\\boxed{{{pred}}}"
#     return pred, cot


# ============================================================
# Bit Solver（DFS trace 版本，训练数据质量更高）
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
# Symbol Solver（来自 solvers.py）
# ============================================================

def parse_numeric_expr(x):
    x = str(x).strip()
    if len(x) != 5: return None
    left, op, right = x[:2], x[2], x[3:]
    if not (left.isdigit() and right.isdigit()): return None
    return {
        "x": x, "left": left, "op": op, "right": right,
        "a": int(left), "b": int(right),
        "ar": int(left[::-1]), "br": int(right[::-1]),
        "l0": int(left[0]), "l1": int(left[1]),
        "r0": int(right[0]), "r1": int(right[1]),
    }

def rev_num_string(s):
    s = str(s)
    return "-" + s[1:][::-1] if s.startswith("-") else s[::-1]

def numeric_candidate_outputs(x):
    p = parse_numeric_expr(x)
    if p is None: return {}
    a, b, ar, br = p["a"], p["b"], p["ar"], p["br"]
    l0, l1, r0, r1, op = p["l0"], p["l1"], p["r0"], p["r1"], p["op"]
    c = {}

    def av(name, s):
        s = str(s)
        c[name] = s
        c[name+"__revout"] = rev_num_string(s)
        c[name+"__op_prefix"] = op+s
        c[name+"__op_suffix"] = s+op
        rev = rev_num_string(s)
        c[name+"__revout_op_prefix"] = op+rev
        c[name+"__revout_op_suffix"] = rev+op

    def add(n, v): av(n, str(v))
    def apm(n, v): add(n,v); add(n+"_plus1",v+1); add(n+"_minus1",v-1)

    apm("a_plus_b", a+b); apm("a_minus_b", a-b); apm("b_minus_a", b-a)
    apm("abs_a_minus_b", abs(a-b)); apm("a_mul_b", a*b)
    apm("ar_plus_br", ar+br); apm("ar_minus_br", ar-br); apm("br_minus_ar", br-ar)
    apm("abs_ar_minus_br", abs(ar-br)); apm("ar_mul_br", ar*br)
    apm("sum_all_digits", l0+l1+r0+r1); apm("sum_left_digits", l0+l1)
    apm("sum_right_digits", r0+r1); apm("abs_sum_digits_diff", abs((l0+l1)-(r0+r1)))
    apm("prod_left_digits", l0*l1); apm("prod_right_digits", r0*r1)
    apm("sum_digit_products_parallel", l0*r0+l1*r1)
    apm("sum_digit_products_cross", l0*r1+l1*r0)

    for name, s in {
        "digit_add_parallel": str(l0+r0)+str(l1+r1),
        "digit_add_cross": str(l0+r1)+str(l1+r0),
        "digit_absdiff_parallel": str(abs(l0-r0))+str(abs(l1-r1)),
        "digit_absdiff_cross": str(abs(l0-r1))+str(abs(l1-r0)),
        "digit_mul_parallel": str(l0*r0)+str(l1*r1),
        "digit_mul_cross": str(l0*r1)+str(l1*r0),
        "digit_sum_then_absdiff": str(l0+l1)+str(abs(r0-r1)),
        "digit_absdiff_then_sum": str(abs(l0-l1))+str(r0+r1),
        "digit_parallel_pairs": str(l0)+str(r0)+str(l1)+str(r1),
        "digit_cross_pairs": str(l0)+str(r1)+str(l1)+str(r0),
        "digit_reverse_parallel_pairs": str(r0)+str(l0)+str(r1)+str(l1),
        "digit_reverse_cross_pairs": str(r1)+str(l0)+str(r0)+str(l1),
    }.items():
        av(name, s)

    for name, s in {
        "left_right": p["left"]+p["right"],
        "right_left": p["right"]+p["left"],
        "rev_left_right": p["left"][::-1]+p["right"],
        "left_rev_right": p["left"]+p["right"][::-1],
        "rev_left_rev_right": p["left"][::-1]+p["right"][::-1],
        "left_right_revout": (p["left"]+p["right"])[::-1],
        "right_left_revout": (p["right"]+p["left"])[::-1],
    }.items():
        av(name, s)

    return c

def score_numeric_rules(examples):
    scores = {}
    for x, y in examples:
        for name, value in numeric_candidate_outputs(x).items():
            if str(value) == str(y):
                scores[name] = scores.get(name, 0) + 1
    return scores

def find_best_numeric_rules(examples):
    scores = score_numeric_rules(examples)
    if not scores: return set()
    best = max(scores.values())
    return {r for r, s in scores.items() if s == best}

def choose_numeric_rule(possible):
    if not possible: return None
    priority = [
        "ar_plus_br__revout","ar_mul_br__revout","left_right",
        "rev_left_rev_right__revout","ar_mul_br_minus1__revout",
        "ar_plus_br_minus1__revout","ar_mul_br_plus1__revout",
        "ar_plus_br_plus1__revout","ar_minus_br__revout",
        "abs_a_minus_b","a_plus_b_plus1","a_plus_b_minus1",
        "a_mul_b_plus1","a_mul_b_minus1","a_mul_b","a_plus_b",
        "abs_ar_minus_br__revout_op_prefix","abs_ar_minus_br__revout",
        "a_minus_b","abs_a_minus_b__revout_op_prefix",
        "abs_ar_minus_br__revout_op_suffix",
        "a_minus_b__op_prefix","a_minus_b__op_suffix",
    ]
    for r in priority:
        if r in possible: return r
    return sorted(possible)[0]

def solve_numeric_symbol_prompt(prompt):
    pairs = []
    for line in prompt.splitlines():
        line = line.strip()
        if " = " in line and not line.startswith("In "):
            x, y = line.split(" = ", 1)
            pairs.append((x.strip(), y.strip()))
    m = re.search(r"result for: (.*)", prompt)
    if not m: return ""
    query_x = m.group(1).strip()
    qi = parse_numeric_expr(query_x)
    if qi is None: return ""
    query_op = qi["op"]
    same_op = [(x,y) for x,y in pairs if parse_numeric_expr(x) and parse_numeric_expr(x)["op"]==query_op]
    used = same_op if same_op else [(x,y) for x,y in pairs if parse_numeric_expr(x)]
    if not used: return ""
    rule = choose_numeric_rule(find_best_numeric_rules(used))
    if rule is None: return ""
    return numeric_candidate_outputs(query_x).get(rule, "")

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

def solve_char_symbol_prompt(prompt):
    pairs = []
    for line in prompt.splitlines():
        line = line.strip()
        if " = " in line and not line.startswith("In "):
            x, y = line.split(" = ", 1)
            pairs.append((x.strip(), y.strip()))
    m = re.search(r"result for: (.*)", prompt)
    if not m: return ""
    target = m.group(1).strip()
    if len(target) != 5: return ""
    target_op = target[2]
    same_op = [(e,o) for e,o in pairs if len(e)==5 and e[2]==target_op]
    examples = same_op if same_op else [(e,o) for e,o in pairs if len(e)==5]
    scores = {}
    for expr, out in examples:
        for name, value in char_symbol_candidates(expr).items():
            if value == out:
                scores[name] = scores.get(name, 0) + 1
    if not scores: return ""
    best = max(scores.values())
    possible = [n for n, s in scores.items() if s == best]
    return char_symbol_candidates(target).get(sorted(possible)[0], "")

def solve_symbol(prompt: str) -> tuple:
    pred = solve_numeric_symbol_prompt(prompt) or solve_char_symbol_prompt(prompt)
    if not pred:
        return "", "No rule found"

    # 重建推理过程
    pairs = []
    for line in prompt.splitlines():
        line = line.strip()
        if " = " in line and not line.startswith("In "):
            x, y = line.split(" = ", 1)
            pairs.append((x.strip(), y.strip()))

    m = re.search(r"result for: (.*)", prompt)
    query_x = m.group(1).strip() if m else ""

    cot = "Step 1: Parse examples:\n"
    for x, y in pairs:
        cot += f"  {x} = {y}\n"
    cot += f"\nStep 2: Scan candidate rules against examples with same operator as query.\n"
    cot += f"\nStep 3: Apply best matching rule to query '{query_x}'.\n"
    cot += f"\nResult: {pred}\n\\boxed{{{pred}}}"
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
# Prompt（symbol 兜底用）
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
2. CRACK cipher: build symbol->digit mapping.
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
# API 调用（symbol 兜底用）
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
    elif task_type in {"symbol_numeric", "symbol_char"}:
        pred, cot = solve_symbol(prompt)
        if not pred:
            system_msg = PROMPTS.get(task_type, PROMPTS["unknown"])
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
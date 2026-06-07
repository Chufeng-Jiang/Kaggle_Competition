import subprocess
import tempfile
import os
import re

PICAT_PATH = os.path.expanduser("~/Desktop/nemotron_reasoning/Picat/picat")


def parse_prompt(prompt: str):
    lines = prompt.strip().splitlines()
    examples = []
    query = None
    for line in lines:
        line = line.strip()
        if line.startswith("Now, determine"):
            m = re.search(r"result for:\s*(.+)", line)
            if m:
                query = m.group(1).strip()
        elif " = " in line and not line.startswith("In "):
            parts = line.split(" = ", 1)
            if len(parts) == 2:
                examples.append((parts[0].strip(), parts[1].strip()))
    return examples, query


def detect_operator_pos(examples):
    return 2


def collect_symbols(examples, op_pos):
    all_symbols = set()
    for inp, out in examples:
        for i, ch in enumerate(inp):
            if i != op_pos:
                all_symbols.add(ch)
        for ch in out:
            all_symbols.add(ch)
    return sorted(all_symbols)


def generate_picat_code(examples, query, op_pos, symbols, sym_idx):
    n = len(symbols)
    var_names = [f"V{i+1}" for i in range(n)]
    lines = []
    lines.append("import cp.")
    lines.append("")
    lines.append("main =>")
    lines.append(f"    Vars = {var_names},")
    lines.append(f"    Vars :: 0..9,")
    lines.append(f"    all_different(Vars),")
    lines.append("")

    def sym_to_var(s):
        return f"V{sym_idx[s]}"

    def build_num_expr(syms):
        if len(syms) == 1:
            return sym_to_var(syms[0]), sym_to_var(syms[0])
        elif len(syms) == 2:
            fwd = f"(10*{sym_to_var(syms[0])} + {sym_to_var(syms[1])})"
            rev = f"(10*{sym_to_var(syms[1])} + {sym_to_var(syms[0])})"
            return fwd, rev
        return None, None

    for ex_idx, (inp, out) in enumerate(examples):
        left_syms = [inp[i] for i in range(op_pos) if inp[i] in sym_idx]
        right_syms = [inp[i] for i in range(op_pos+1, len(inp)) if inp[i] in sym_idx]
        out_syms = [ch for ch in out if ch in sym_idx]

        if not left_syms or not right_syms or not out_syms:
            continue

        left_fwd, left_rev = build_num_expr(left_syms)
        right_fwd, right_rev = build_num_expr(right_syms)

        if len(out_syms) == 1:
            out_expr = sym_to_var(out_syms[0])
        elif len(out_syms) == 2:
            out_expr = f"(10*{sym_to_var(out_syms[0])} + {sym_to_var(out_syms[1])})"
        else:
            continue

        lines.append(f"    % Example {ex_idx+1}: {inp} = {out}")
        lines.append(f"    ({left_fwd} + {right_fwd} =:= {out_expr}")
        lines.append(f"    ; {left_rev} + {right_rev} =:= {out_expr}")
        lines.append(f"    ; {left_fwd} * {right_fwd} =:= {out_expr}")
        lines.append(f"    ; {left_rev} * {right_rev} =:= {out_expr}")
        lines.append(f"    ; {left_fwd} - {right_fwd} =:= {out_expr}")
        lines.append(f"    ; {right_fwd} - {left_fwd} =:= {out_expr}")
        lines.append(f"    ; {left_rev} - {right_rev} =:= {out_expr}")
        lines.append(f"    ; {right_rev} - {left_rev} =:= {out_expr}),")
        lines.append("")

    lines.append(f"    solve(Vars),")
    lines.append("")
    for i, sym in enumerate(symbols):
        lines.append(f"    writeln(sym_{i+1} = V{i+1}),")
    lines.append("    halt.")
    return "\n".join(lines)


def parse_picat_output(output: str, symbols: list) -> dict:
    mapping = {}
    for i, sym in enumerate(symbols):
        m = re.search(rf"sym_{i+1} = (\d+)", output)
        if m:
            mapping[sym] = int(m.group(1))
    return mapping


def apply_combo(left_str, right_str, op_mode, op_func, fmt_func):
    try:
        if op_mode == "AB_CD":
            a, b = int(left_str), int(right_str)
        elif op_mode == "BA_DC":
            a, b = int(left_str[::-1]), int(right_str[::-1])
        elif op_mode == "AB_DC":
            a, b = int(left_str), int(right_str[::-1])
        elif op_mode == "BA_CD":
            a, b = int(left_str[::-1]), int(right_str)
        result = op_func(a, b)
        return fmt_func(result)
    except:
        return None


def get_scan_order():
    op_modes = ["BA_DC", "AB_CD", "AB_DC", "BA_CD"]
    operations = [
        ("add",          lambda a, b: a + b),
        ("mul",          lambda a, b: a * b),
        ("sub",          lambda a, b: a - b),
        ("rsub",         lambda a, b: b - a),
        ("add1",         lambda a, b: a + b + 1),
        ("sub1",         lambda a, b: a - b - 1),
        ("mul_add1",     lambda a, b: a * b + 1),
        ("mul_sub1",     lambda a, b: a * b - 1),
        ("max",          lambda a, b: max(a, b)),
        ("min",          lambda a, b: min(a, b)),
        ("max_mod_min",  lambda a, b: max(a,b) % min(a,b) if min(a,b) != 0 else max(a,b)),
        ("abs_diff",     lambda a, b: abs(a - b)),
        ("cat",          lambda a, b: int(str(a) + str(b))),
        ("rcat",         lambda a, b: int(str(b) + str(a))),
    ]
    formats = [
        ("raw",   lambda x: str(x)),
        ("rev",   lambda x: str(x)[::-1]),
        ("abs",   lambda x: str(abs(x))),
        ("zpad2", lambda x: str(x).zfill(2)),
    ]
    return op_modes, operations, formats


def solve_cipher_digit(prompt: str) -> str:
    examples, query = parse_prompt(prompt)
    if not examples or not query:
        return ""

    op_pos = detect_operator_pos(examples)
    if op_pos is None:
        return ""

    symbols = collect_symbols(examples, op_pos)
    sym_idx = {s: i+1 for i, s in enumerate(symbols)}

    if len(symbols) > 10:
        return ""

    # Step 1: 用Picat找符号->数字映射
    picat_code = generate_picat_code(examples, query, op_pos, symbols, sym_idx)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.pi', delete=False) as f:
        f.write(picat_code)
        fname = f.name

    try:
        result = subprocess.run(
            [PICAT_PATH, fname],
            capture_output=True, text=True, timeout=10
        )
        mapping = parse_picat_output(result.stdout, symbols)
    finally:
        os.unlink(fname)

    if not mapping or len(mapping) != len(symbols):
        return ""

    digit_to_sym = {v: k for k, v in mapping.items()}

    # Step 2: 解码所有例子
    decoded = []
    for inp, out in examples:
        left = "".join(str(mapping[ch]) for i, ch in enumerate(inp) if i < op_pos and ch in mapping)
        right = "".join(str(mapping[ch]) for i, ch in enumerate(inp) if i > op_pos and ch in mapping)
        out_dec = "".join(str(mapping.get(ch, "?")) for ch in out)
        if "?" in out_dec:
            return ""
        decoded.append((left, right, out_dec))

    # Step 3: 扫描操作组合
    op_modes, operations, formats = get_scan_order()
    found_rule = None
    for op_mode in op_modes:
        for op_name, op_func in operations:
            for fmt_name, fmt_func in formats:
                if all(apply_combo(l, r, op_mode, op_func, fmt_func) == e for l, r, e in decoded):
                    found_rule = (op_mode, op_name, fmt_name, op_func, fmt_func)
                    break
            if found_rule: break
        if found_rule: break

    if not found_rule:
        return ""

    op_mode, op_name, fmt_name, op_func, fmt_func = found_rule

    # Step 4: 应用到query
    q_left = "".join(str(mapping[ch]) for i, ch in enumerate(query) if i < op_pos and ch in mapping)
    q_right = "".join(str(mapping[ch]) for i, ch in enumerate(query) if i > op_pos and ch in mapping)
    pred_raw = apply_combo(q_left, q_right, op_mode, op_func, fmt_func)
    if pred_raw is None:
        return ""

    # Step 5: 编码回符号
    pred_sym = "".join(digit_to_sym.get(int(d), "?") for d in pred_raw)
    return pred_sym


# ============================================================
# 测试
# ============================================================
if __name__ == "__main__":
    import pandas as pd
    import sys
    sys.path.insert(0, "src")
    from generate_cot import normalize_task_type

    df = pd.read_csv("data/train.csv", dtype=str, keep_default_na=False)
    df["task_type"] = df.apply(lambda row: normalize_task_type(row.to_dict()), axis=1)
    symbol_char = df[df["task_type"] == "symbol_char"].head(20)

    print(f"symbol_char count: {len(symbol_char)}")
    correct = 0
    total = 0
    for _, row in symbol_char.iterrows():
        pred = solve_cipher_digit(row["prompt"])
        gt = row["answer"]
        is_correct = (pred == gt)
        if is_correct:
            correct += 1
        total += 1
        print(f"GT: {gt!r:15s} PRED: {pred!r:15s} {'✓' if is_correct else '✗'}")

    print(f"\nAccuracy: {correct}/{total} ({correct/total:.1%})")
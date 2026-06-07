"""cipher_digit_solver.py"""

import re
from itertools import permutations

def parse_prompt(prompt: str):
    """解析prompt，返回examples和query"""
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
    """输入固定5字符，操作符在位置2"""
    if not examples:
        return None
    
    # 验证所有输入长度是否为5
    input_lens = [len(ex[0]) for ex in examples]
    if all(l == 5 for l in input_lens):
        return 2
    
    # 否则找所有例子中变化最少的位置
    if not examples:
        return None
    input_len = len(examples[0][0])
    min_variety = float('inf')
    best_pos = None
    for pos in range(input_len):
        chars = [ex[0][pos] for ex in examples if len(ex[0]) > pos]
        variety = len(set(chars))
        if variety < min_variety:
            min_variety = variety
            best_pos = pos
    return best_pos


def build_symbol_mapping(examples, op_pos):
    """
    暴力枚举符号->数字的映射
    返回所有可能的映射列表
    """
    # 收集所有出现的符号（排除操作符）
    all_symbols = set()
    for inp, out in examples:
        for i, ch in enumerate(inp):
            if i != op_pos:
                all_symbols.add(ch)
        for ch in out:
            all_symbols.add(ch)
    
    all_symbols = sorted(all_symbols)
    digits = list(range(10))
    
    print(f"symbols: {all_symbols} ({len(all_symbols)} total)")
    
    if len(all_symbols) > 10:
        print("Too many symbols, cannot map to digits 0-9")
        return []
    
    # 枚举所有可能的符号->数字映射
    valid_mappings = []
    
    for digit_perm in permutations(digits, len(all_symbols)):
        mapping = dict(zip(all_symbols, digit_perm))
        
        # 检查这个映射是否能让所有例子解码出合理的数字
        valid = True
        decoded_examples = []
        
        for inp, out in examples:
            left = ""
            right = ""
            for i, ch in enumerate(inp):
                if i < op_pos:
                    if ch not in mapping:
                        valid = False; break
                    left += str(mapping[ch])
                elif i > op_pos:
                    if ch not in mapping:
                        valid = False; break
                    right += str(mapping[ch])
            
            if not valid:
                break
                
            out_decoded = ""
            for ch in out:
                if ch not in mapping:
                    valid = False; break
                out_decoded += str(mapping[ch])
            
            if not valid:
                break
                
            decoded_examples.append((left, right, out_decoded))
        
        if valid:
            valid_mappings.append((mapping, decoded_examples))
    
    return valid_mappings


# 47种常见操作组合（按频率排序）
def get_scan_order():
    operand_modes = ["BA_DC", "AB_CD", "AB_DC", "BA_CD"]
    
    operations = [
        ("add",    lambda a, b: a + b),
        ("mul",    lambda a, b: a * b),
        ("sub",    lambda a, b: a - b),
        ("rsub",   lambda a, b: b - a),
        ("add1",   lambda a, b: a + b + 1),
        ("sub1",   lambda a, b: a - b - 1),
        ("mul_add1", lambda a, b: a * b + 1),
        ("mul_sub1", lambda a, b: a * b - 1),
        ("max",    lambda a, b: max(a, b)),
        ("min",    lambda a, b: min(a, b)),
        ("max_mod_min", lambda a, b: max(a,b) % min(a,b) if min(a,b) != 0 else max(a,b)),
        ("abs_diff", lambda a, b: abs(a - b)),
        ("cat",    lambda a, b: int(str(a) + str(b))),
        ("rcat",   lambda a, b: int(str(b) + str(a))),
    ]
    
    formats = [
        ("raw",   lambda x: str(x)),
        ("rev",   lambda x: str(x)[::-1]),
        ("abs",   lambda x: str(abs(x))),
        ("zpad2", lambda x: str(x).zfill(2)),
    ]
    
    return operand_modes, operations, formats


def apply_combo(left_str, right_str, op_mode, op_func, fmt_func):
    """应用一种组合，返回结果字符串"""
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


def scan_operations(decoded_examples):
    """
    扫描所有操作组合，找到能解释所有例子的那个
    返回 (op_mode, op_name, fmt_name, op_func, fmt_func) 或 None
    """
    op_modes, operations, formats = get_scan_order()
    
    for op_mode in op_modes:
        for op_name, op_func in operations:
            for fmt_name, fmt_func in formats:
                # 检查是否所有例子都满足
                all_match = True
                for left, right, expected_out in decoded_examples:
                    result = apply_combo(left, right, op_mode, op_func, fmt_func)
                    if result != expected_out:
                        all_match = False
                        break
                
                if all_match:
                    return op_mode, op_name, fmt_name, op_func, fmt_func
    
    return None


def solve_cipher_digit(prompt: str) -> str:
    examples, query = parse_prompt(prompt)
    
    if not examples or not query:
        print("Parse failed")
        return ""
    
    print(f"Examples: {examples}")
    print(f"Query: {query}")
    
    # 找操作符位置
    op_pos = detect_operator_pos(examples)
    if op_pos is None:
        print("Cannot detect operator position")
        return ""
    
    op_symbol = examples[0][0][op_pos]
    print(f"Operator: '{op_symbol}' at position {op_pos}")
    
    # 枚举符号->数字映射
    print("Building symbol mappings...")
    valid_mappings = build_symbol_mapping(examples, op_pos)
    print(f"Found {len(valid_mappings)} valid mappings")
    
    # 对每个映射，扫描操作
    solutions = []
    for mapping, decoded_examples in valid_mappings:
        result = scan_operations(decoded_examples)
        if result:
            op_mode, op_name, fmt_name, op_func, fmt_func = result
            
            # 应用到query
            query_left = ""
            query_right = ""
            for i, ch in enumerate(query):
                if i < op_pos:
                    query_left += str(mapping.get(ch, "?"))
                elif i > op_pos:
                    query_right += str(mapping.get(ch, "?"))
            
            pred_raw = apply_combo(query_left, query_right, op_mode, op_func, fmt_func)
            if pred_raw is None:
                continue
            
            # 把数字结果编码回符号
            digit_to_sym = {str(v): k for k, v in mapping.items()}
            pred_sym = "".join(digit_to_sym.get(d, "?") for d in pred_raw)
            
            solutions.append({
                "mapping": mapping,
                "op_mode": op_mode,
                "op_name": op_name,
                "fmt_name": fmt_name,
                "pred_raw": pred_raw,
                "pred_sym": pred_sym,
            })
    
    print(f"\nFound {len(solutions)} solutions:")
    for sol in solutions[:5]:
        print(f"  mapping={sol['mapping']}")
        print(f"  rule: {sol['op_mode']} | {sol['op_name']} | {sol['fmt_name']}")
        print(f"  pred: {sol['pred_raw']} -> '{sol['pred_sym']}'")
        print()
    
    if solutions:
        return solutions[0]["pred_sym"]
    return ""


# ============================================================
# 测试
# ============================================================
if __name__ == "__main__":
    import pandas as pd
    
    df = pd.read_csv("data/train.csv", dtype=str, keep_default_na=False)
    
    # 过滤 symbol_char
    from generate_cot import normalize_task_type
    df["task_type"] = df.apply(lambda row: normalize_task_type(row.to_dict()), axis=1)
    symbol_char = df[df["task_type"] == "symbol_char"].head(5)
    
    print(f"symbol_char count: {len(symbol_char)}")
    
    for _, row in symbol_char.iterrows():
        print("=" * 60)
        print(f"GT: {row['answer']}")
        pred = solve_cipher_digit(row["prompt"])
        print(f"PRED: {pred}")
        print(f"CORRECT: {pred == row['answer']}")
        print()
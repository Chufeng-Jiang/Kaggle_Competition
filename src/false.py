import json
import pandas as pd
import re
from itertools import permutations

with open("outputs/checkpoint.json", "r", encoding="utf-8") as f:
    results = json.load(f)

df = pd.DataFrame(results)
symbol = df[df["type"] == "symbol_char"].copy()
total = len(symbol)
print(f"Total symbol_char: {total}")

def parse_examples(prompt):
    lines = prompt.strip().splitlines()
    examples = []
    query = None
    for line in lines:
        line = line.strip()
        if line.startswith("Now, determine"):
            m = re.search(r"result for:\s*(.+)", line)
            if m:
                query = m.group(1).strip()
        elif " = " in line:
            parts = line.split(" = ", 1)
            if len(parts) == 2:
                examples.append((parts[0].strip(), parts[1].strip()))
    return examples, query

def classify_topic(examples):
    if not examples:
        return "unknown"
    if all(set(out).issubset(set(inp)) for inp, out in examples):
        return "deletion"
    total_out = sum(len(out) for _, out in examples)
    if total_out == 0:
        return "unknown"
    new_char_ratio = sum(
        sum(1 for c in out if c not in inp)
        for inp, out in examples
    ) / total_out
    if new_char_ratio > 0.3:
        return "mapping"
    return "mixed"

# 对每道mapping类题，分析only_left/only_right和输出的关系
mapping_rows = []
for _, row in symbol.iterrows():
    examples, query = parse_examples(row["prompt"])
    gt = row["answer"]
    topic = classify_topic(examples)
    if topic != "mapping":
        continue

    # 找操作符位置（假设输入长度固定为5，操作符在中间某位）
    # 尝试找一个固定位置的字符在所有examples里都相同
    op_pos = None
    for pos in range(5):
        chars_at_pos = [inp[pos] for inp, out in examples if len(inp) > pos]
        if len(set(chars_at_pos)) == 1:
            op_pos = pos
            break

    mapping_rows.append({
        "prompt": row["prompt"],
        "query": query,
        "gt": gt,
        "op_pos": op_pos,
        "examples": examples,
        "topic": topic,
    })

print(f"\nMapping类总数: {len(mapping_rows)}")

# 统计op_pos分布
op_pos_counts = pd.Series([r["op_pos"] for r in mapping_rows]).value_counts()
print(f"\n操作符位置分布:\n{op_pos_counts}")

# 打印几道题详细分析
print("\n=== 详细分析前5道mapping类题 ===")
for r in mapping_rows[:5]:
    print(f"op_pos={r['op_pos']}")
    for inp, out in r["examples"]:
        if r["op_pos"] is not None:
            left = inp[:r["op_pos"]]
            op = inp[r["op_pos"]]
            right = inp[r["op_pos"]+1:]
        else:
            left = inp
            op = "?"
            right = ""
        only_left = [c for c in left if c not in right]
        only_right = [c for c in right if c not in left]
        print(f"  {inp} -> {out}  |  left={left} op={op} right={right}  only_left={only_left} only_right={only_right}")
    print(f"  query={r['query']} GT={r['gt']}")
    print()
    
    
def try_char_mapping_with_deletion(examples):
    """
    尝试建立字符->字符的映射，允许某些字符映射到空字符串
    前提：输出是输入字符按顺序映射后拼接
    """
    from itertools import product
    
    mapping = {}
    
    for inp, out in examples:
        # 尝试把out分配给inp的每个字符
        # 用动态规划：inp[i]映射到out[j:k]
        n, m = len(inp), len(out)
        # 简化：先假设每个字符映射到0或1个字符
        # 枚举哪些位置映射到字符，哪些映射到空
        found = False
        for mask in range(1 << n):
            # mask的第i位=1表示inp[i]映射到一个字符，=0表示映射到空
            if bin(mask).count('1') != m:
                continue
            # 建立这个分配下的映射
            local_map = {}
            out_idx = 0
            valid = True
            for i in range(n):
                c = inp[i]
                if mask & (1 << i):
                    mapped = out[out_idx]
                    out_idx += 1
                else:
                    mapped = ""
                if c in local_map and local_map[c] != mapped:
                    valid = False
                    break
                local_map[c] = mapped
            if valid:
                # 检查是否和已有映射冲突
                conflict = False
                for c, v in local_map.items():
                    if c in mapping and mapping[c] != v:
                        conflict = True
                        break
                if not conflict:
                    mapping.update(local_map)
                    found = True
                    break
        if not found:
            return None
    
    return mapping

for r in mapping_rows[:3]:
    print("=== EXAMPLES ===")
    for inp, out in r["examples"]:
        # 统计重复字符
        from collections import Counter
        inp_counter = Counter(inp)
        repeats = {c: n for c, n in inp_counter.items() if n > 1}
        print(f"  {inp} -> {out}  重复字符: {repeats}  输出长度: {len(out)}")
    print(f"  query={r['query']} GT={r['gt']}")
    print()
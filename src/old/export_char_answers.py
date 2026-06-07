import re
import pandas as pd

from old.parser import detect_task_type


def extract_query(prompt):
    m = re.search(r"result for: (.*)", prompt)
    return m.group(1).strip() if m else ""


def is_char_expr(x):
    x = str(x)

    if len(x) != 5:
        return False

    return not (
        x[:2].isdigit()
        and x[3:].isdigit()
    )


df = pd.read_csv("data/train.csv")

rows = []

for _, row in df.iterrows():

    if detect_task_type(row["prompt"]) != "symbol":
        continue

    query = extract_query(row["prompt"])

    if not is_char_expr(query):
        continue

    answer = str(row["answer"])

    rows.append({
        "task_id": row["id"],
        "query": query,
        "answer": answer,
        "answer_len": len(answer),
        "answer_sorted": "".join(sorted(answer)),
        "answer_charset": "".join(sorted(set(answer))),
    })

out = pd.DataFrame(rows)

out = out.sort_values(
    ["answer_len", "answer_sorted", "answer"]
)

out.to_csv(
    "outputs/symbol_char_answers_sorted.csv",
    index=False
)

print("char tasks:", len(out))

print("\nLength distribution:")
print(
    out["answer_len"]
    .value_counts()
    .sort_index()
)

print("\nTop answers:")
print(
    out["answer"]
    .value_counts()
    .head(50)
)

print(
    "\nsaved outputs/symbol_char_answers_sorted.csv"
)
import re
import pandas as pd

from old.parser import detect_task_type


def extract_pairs(prompt: str):
    pairs = []

    for line in prompt.splitlines():
        line = line.strip()

        if " = " in line and not line.startswith("In "):
            x, y = line.split(" = ", 1)
            pairs.append((x.strip(), y.strip()))

    return pairs


def extract_query(prompt: str):
    m = re.search(r"result for: (.*)", prompt)
    if not m:
        return ""
    return m.group(1).strip()


def is_char_expr(x: str):
    x = str(x)

    if len(x) != 5:
        return False

    left = x[:2]
    right = x[3:]

    return not (left.isdigit() and right.isdigit())


def main():
    df = pd.read_csv("data/train.csv")

    rows = []

    for _, row in df.iterrows():
        prompt = row["prompt"]

        if detect_task_type(prompt) != "symbol":
            continue

        query = extract_query(prompt)

        if not is_char_expr(query):
            continue

        pairs = extract_pairs(prompt)

        char_pairs = [
            (x, y)
            for x, y in pairs
            if is_char_expr(x)
        ]

        if not char_pairs:
            continue

        examples_text = "\n".join(
            f"{x} -> {y}"
            for x, y in char_pairs
        )

        input_text = (
            "Examples:\n"
            + examples_text
            + "\nQuery:\n"
            + query
            + "\nAnswer:"
        )

        rows.append({
            "task_id": row["id"],
            "query": query,
            "answer": str(row["answer"]),
            "answer_len": len(str(row["answer"])),
            "num_examples": len(char_pairs),

            # compact structured fields
            "examples": examples_text,
            "example_inputs": " ||| ".join(x for x, _ in char_pairs),
            "example_outputs": " ||| ".join(y for _, y in char_pairs),

            # model-ready text
            "input_text": input_text,
            "target_text": str(row["answer"]),

            # original full prompt
            "prompt": prompt,
        })

    out = pd.DataFrame(rows)

    out.to_csv("outputs/symbol_char_task_level.csv", index=False)

    print("saved outputs/symbol_char_task_level.csv")
    print("rows:", len(out))

    print("\nLength distribution:")
    print(out["answer_len"].value_counts().sort_index())

    print("\nExample:")
    print(out.head(5)[[
        "task_id",
        "query",
        "answer",
        "num_examples",
        "examples",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
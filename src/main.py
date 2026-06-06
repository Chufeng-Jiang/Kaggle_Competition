"""main.py"""
import pandas as pd
from tqdm import tqdm

from parser import detect_task_type
from solvers import (
    solve_roman,
    solve_gravity,
    solve_unit,
    solve_cipher,
    solve_symbol,
)


def solve(prompt: str, vocab: set) -> str:
    task_type = detect_task_type(prompt)

    if task_type == "roman":
        return solve_roman(prompt)

    if task_type == "gravity":
        return solve_gravity(prompt)

    if task_type == "unit":
        return solve_unit(prompt)

    if task_type == "cipher":
        return solve_cipher(prompt, vocab)
    
    if task_type == "symbol":
        return solve_symbol(prompt)

    return ""


def normalize(x):
    s = str(x).strip()

    # 如果是数字，统一成 float 再比较
    try:
        v = float(s)
        return f"{v:.2f}"
    except:
        return s


def main():
    df = pd.read_csv("data/train.csv")
    
    cipher_answers = df[
        df["prompt"].str.contains("secret encryption", case=False)
    ]["answer"]

    vocab = set()
    for ans in cipher_answers:
        for word in str(ans).split():
            vocab.add(word)

    task_types = []
    preds = []

    for prompt in tqdm(df["prompt"]):
        task_type = detect_task_type(prompt)
        pred = solve(prompt, vocab)

        task_types.append(task_type)
        preds.append(pred)

    df["task_type"] = task_types
    df["pred"] = preds

    df["correct"] = (
        df["pred"].apply(normalize)
        == df["answer"].apply(normalize)
    )

    print("\nTask counts:")
    print(df["task_type"].value_counts())

    print("\nAccuracy by task:")
    print(df.groupby("task_type")["correct"].mean())

    print("\nOverall accuracy:")
    print(df["correct"].mean())

    df.to_csv("outputs/train_predictions.csv", index=False)


if __name__ == "__main__":
    main()
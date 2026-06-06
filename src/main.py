import pandas as pd
from tqdm import tqdm

from parser import detect_task_type
from solvers import solve_roman, solve_gravity, solve_unit


def solve(prompt: str) -> str:
    task_type = detect_task_type(prompt)

    if task_type == "roman":
        return solve_roman(prompt)

    if task_type == "gravity":
        return solve_gravity(prompt)

    if task_type == "unit":
        return solve_unit(prompt)

    return ""


def normalize(x):
    return str(x).strip()


def main():
    df = pd.read_csv("data/train.csv")

    task_types = []
    preds = []

    for prompt in tqdm(df["prompt"]):
        task_type = detect_task_type(prompt)
        pred = solve(prompt)

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
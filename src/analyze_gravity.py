import re
import pandas as pd
import numpy as np


def normalize(x):
    try:
        return f"{float(str(x).strip()):.2f}"
    except:
        return str(x).strip()


def format_number(x):
    x = round(x + 1e-12, 2)
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return f"{x:.2f}"


def parse_gravity(prompt):
    examples = re.findall(
        r"For t = ([\d.]+)s, distance = ([\d.]+) m",
        prompt
    )

    target = re.search(
        r"for t = ([\d.]+)s",
        prompt.split("Now")[-1]
    )

    xs = []
    ys = []

    for t, d in examples:
        t = float(t)
        d = float(d)
        xs.append(0.5 * t * t)
        ys.append(d)

    return xs, ys, float(target.group(1))


def fit_g_interval(xs, ys):
    low = -1e18
    high = 1e18

    for x, y in zip(xs, ys):
        lo = (y - 0.005) / x
        hi = (y + 0.005) / x

        low = max(low, lo)
        high = min(high, hi)

    return low, high


def predict(prompt, alpha=0.5):
    xs, ys, t_new = parse_gravity(prompt)
    low, high = fit_g_interval(xs, ys)

    g = low + alpha * (high - low)

    pred = g * 0.5 * t_new * t_new
    return format_number(pred), low, high, g


def main():
    df = pd.read_csv("data/train.csv")

    df = df[df["prompt"].str.contains("gravitational constant", case=False)].copy()

    rows = []

    for _, row in df.iterrows():
        pred, low, high, g = predict(row["prompt"], alpha=0.5)

        rows.append({
            "id": row["id"],
            "pred": pred,
            "answer": row["answer"],
            "correct": normalize(pred) == normalize(row["answer"]),
            "g_low": low,
            "g_high": high,
            "g_used": g,
            "g_width": high - low,
            "prompt": row["prompt"],
        })

    out = pd.DataFrame(rows)

    print("Gravity accuracy:", out["correct"].mean())

    print("\nWrong examples:")
    print(out[out["correct"] == False][
        ["id", "pred", "answer", "g_low", "g_high", "g_width"]
    ].head(20))

    out.to_csv("outputs/gravity_analysis.csv", index=False)


if __name__ == "__main__":
    main()
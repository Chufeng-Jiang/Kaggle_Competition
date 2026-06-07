import pandas as pd


def main():
    df = pd.read_csv(
        "outputs/bit_ensemble_compare.csv",
        dtype=str
    )

    df["expr_correct"] = (
        df["expr_correct"]
        .str.lower()
        .eq("true")
    )

    rows = []

    for _, r in df.iterrows():

        answer = str(r["answer"]).zfill(8)
        pred = str(r["expr_pred"]).zfill(8)

        diff = [
            i
            for i, (a, b)
            in enumerate(zip(answer, pred))
            if a != b
        ]

        if len(diff) == 1:

            rows.append({
                "id": r["id"],
                "flip_pos": diff[0],
                "answer": answer,
                "expr_pred": pred,
                "byte_pred": r["byte_pred"],
                "chunk_pred": r["chunk_pred"],
                "expr_train_acc": r["expr_train_acc"],
                "byte_rule": r["byte_rule"],
                "chunk_rule": r["chunk_rule"],
            })

    out = pd.DataFrame(rows)

    print("one bit wrong:", len(out))

    print("\nflip position distribution:")
    print(
        out["flip_pos"]
        .value_counts()
        .sort_index()
    )

    out = out.sort_values(
        ["flip_pos", "expr_train_acc"],
        ascending=[True, False]
    )

    out.to_csv(
        "outputs/bit_one_bit_errors.csv",
        index=False
    )

    print(
        "\nsaved outputs/bit_one_bit_errors.csv"
    )


if __name__ == "__main__":
    main()
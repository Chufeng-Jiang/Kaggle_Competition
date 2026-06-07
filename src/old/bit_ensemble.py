import pandas as pd


def main():
    byte_df = pd.read_csv("outputs/bit_search_results.csv", dtype=str)
    expr_df = pd.read_csv("outputs/bit_expr_search_results.csv", dtype=str)
    chunk_df = pd.read_csv("outputs/bit_chunk_search_results.csv", dtype=str)
    
    # restore boolean columns

    byte_df["correct"] = (
        byte_df["correct"]
        .astype(str)
        .str.lower()
        .eq("true")
    )

    expr_df["correct"] = (
        expr_df["correct"]
        .astype(str)
        .str.lower()
        .eq("true")
    )

    chunk_df["correct"] = (
        chunk_df["correct"]
        .astype(str)
        .str.lower()
        .eq("true")
    )

    # restore numeric columns

    byte_df["score"] = pd.to_numeric(
        byte_df["score"],
        errors="coerce"
    ).fillna(0)

    chunk_df["score"] = pd.to_numeric(
        chunk_df["score"],
        errors="coerce"
    ).fillna(0)

    expr_df["train_acc"] = pd.to_numeric(
        expr_df["train_acc"],
        errors="coerce"
    ).fillna(0)

    byte_df = byte_df.rename(columns={
        "pred": "byte_pred",
        "correct": "byte_correct",
        "rule": "byte_rule",
        "score": "byte_score",
    })

    expr_df = expr_df.rename(columns={
        "pred": "expr_pred",
        "correct": "expr_correct",
        "rule": "expr_rule",
        "train_acc": "expr_train_acc",
    })

    chunk_df = chunk_df.rename(columns={
        "pred": "chunk_pred",
        "correct": "chunk_correct",
        "rule": "chunk_rule",
        "score": "chunk_score",
    })

    df = byte_df.merge(
        expr_df[["id", "expr_pred", "expr_correct", "expr_rule", "expr_train_acc"]],
        on="id",
        how="inner",
    ).merge(
        chunk_df[["id", "chunk_pred", "chunk_correct", "chunk_rule", "chunk_score"]],
        on="id",
        how="inner",
    )

    print("bit tasks:", len(df))
    print("byte acc:", df["byte_correct"].mean())
    print("expr acc:", df["expr_correct"].mean())
    print("chunk acc:", df["chunk_correct"].mean())

    oracle = (
        df["byte_correct"]
        | df["expr_correct"]
        | df["chunk_correct"]
    ).mean()

    print("\noracle byte OR expr OR chunk:", oracle)

    print("\nOverlap counts:")
    print("byte_only:", ((df["byte_correct"]) & (~df["expr_correct"]) & (~df["chunk_correct"])).sum())
    print("expr_only:", ((~df["byte_correct"]) & (df["expr_correct"]) & (~df["chunk_correct"])).sum())
    print("chunk_only:", ((~df["byte_correct"]) & (~df["expr_correct"]) & (df["chunk_correct"])).sum())

    best_acc = 0
    best_rule = None

    print("\nTry thresholds:")

    for byte_t in range(1, 11):
        for chunk_t in range(1, 11):
            preds = []

            for _, r in df.iterrows():
                if r["byte_score"] >= byte_t:
                    preds.append(r["byte_pred"])
                elif r["chunk_score"] >= chunk_t:
                    preds.append(r["chunk_pred"])
                else:
                    preds.append(r["expr_pred"])

            acc = (
                pd.Series(preds).astype(str).values
                == df["answer"].astype(str).values
            ).mean()

            if acc > best_acc:
                best_acc = acc
                best_rule = (byte_t, chunk_t)

    print("best byte_t, chunk_t:", best_rule)
    print("best acc:", best_acc)

    print("\nchunk_only examples:")
    print(
        df[
            (~df["byte_correct"])
            & (~df["expr_correct"])
            & (df["chunk_correct"])
        ][
            [
                "id",
                "answer",
                "byte_pred",
                "byte_rule",
                "byte_score",
                "expr_pred",
                "expr_train_acc",
                "chunk_pred",
                "chunk_rule",
                "chunk_score",
            ]
        ].head(40).to_string(index=False)
    )

    df.to_csv("outputs/bit_ensemble_compare.csv", index=False)
    print("\nsaved outputs/bit_ensemble_compare.csv")


if __name__ == "__main__":
    main()
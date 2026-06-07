import pandas as pd


def hamming(a, b):
    return sum(x != y for x, y in zip(a, b))


def main():
    df = pd.read_csv("outputs/bit_ensemble_compare.csv", dtype=str)

    # Use expr as main baseline
    df["expr_correct"] = df["expr_correct"].str.lower().eq("true")
    df["byte_correct"] = df["byte_correct"].str.lower().eq("true")
    df["chunk_correct"] = df["chunk_correct"].str.lower().eq("true")

    rows = []

    for _, r in df.iterrows():
        ans = str(r["answer"]).zfill(8)
        expr = str(r["expr_pred"]).zfill(8)
        byte = str(r["byte_pred"]).zfill(8)
        chunk = str(r["chunk_pred"]).zfill(8)

        if len(ans) != 8 or len(expr) != 8:
            continue

        diff_pos = [
            i for i, (a, p) in enumerate(zip(ans, expr))
            if a != p
        ]

        rows.append({
            "id": r["id"],
            "answer": ans,
            "expr_pred": expr,
            "byte_pred": byte,
            "chunk_pred": chunk,
            "expr_correct": r["expr_correct"],
            "byte_correct": r["byte_correct"],
            "chunk_correct": r["chunk_correct"],
            "num_diff": len(diff_pos),
            "diff_pos": ",".join(map(str, diff_pos)),
            "only_msb_wrong": diff_pos == [0],
            "only_lsb_wrong": diff_pos == [7],
            "byte_fixes": byte == ans,
            "chunk_fixes": chunk == ans,
        })

    out = pd.DataFrame(rows)
    out.to_csv("outputs/bit_msb_error_analysis.csv", index=False)

    wrong = out[out["expr_correct"] == False]

    print("total:", len(out))
    print("expr wrong:", len(wrong))

    print("\nDiff count distribution among expr wrong:")
    print(wrong["num_diff"].value_counts().sort_index())

    print("\nOnly MSB wrong:", wrong["only_msb_wrong"].sum())
    print("Only LSB wrong:", wrong["only_lsb_wrong"].sum())

    print("\nPosition error counts:")
    pos_counts = {}
    for s in wrong["diff_pos"]:
        if not isinstance(s, str) or s == "":
            continue
        for p in s.split(","):
            pos_counts[p] = pos_counts.get(p, 0) + 1

    print(
        pd.Series(pos_counts)
        .sort_index()
        .to_string()
    )

    print("\nCases where byte/chunk fixes only-MSB:")
    print(
        wrong[
            (wrong["only_msb_wrong"] == True)
            & ((wrong["byte_fixes"] == True) | (wrong["chunk_fixes"] == True))
        ].head(50).to_string(index=False)
    )

    print("\nTop one-bit wrong examples:")
    print(
        wrong[wrong["num_diff"] == 1]
        .head(80)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
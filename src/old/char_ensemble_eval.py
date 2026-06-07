import pandas as pd


def main():
    template = pd.read_csv("outputs/symbol_char_search_results.csv")
    parser = pd.read_csv("outputs/char_parser_results.csv")

    template = template.rename(columns={
        "pred": "template_pred",
        "correct": "template_correct",
        "rule": "template_rule",
        "score": "template_score",
    })

    parser = parser.rename(columns={
        "pred": "parser_pred",
        "correct": "parser_correct",
        "rule": "parser_rule",
        "score": "parser_score",
    })

    merged = template.merge(
        parser[
            [
                "id",
                "parser_pred",
                "parser_correct",
                "parser_rule",
                "parser_score",
            ]
        ],
        on="id",
        how="inner",
    )

    print("merged:", len(merged))

    print("template acc:", merged["template_correct"].mean())
    print("parser acc:", merged["parser_correct"].mean())

    both_correct = (
        (merged["template_correct"] == True)
        & (merged["parser_correct"] == True)
    ).sum()

    template_only = (
        (merged["template_correct"] == True)
        & (merged["parser_correct"] == False)
    ).sum()

    parser_only = (
        (merged["template_correct"] == False)
        & (merged["parser_correct"] == True)
    ).sum()

    neither = (
        (merged["template_correct"] == False)
        & (merged["parser_correct"] == False)
    ).sum()

    print("\nOverlap:")
    print("both_correct:", both_correct)
    print("template_only:", template_only)
    print("parser_only:", parser_only)
    print("neither:", neither)

    oracle = (
        (merged["template_correct"] == True)
        | (merged["parser_correct"] == True)
    ).mean()

    print("\noracle template OR parser:", oracle)

    print("\nParser-only correct examples:")
    cols = [
        "id",
        "query_x",
        "answer",
        "template_pred",
        "template_rule",
        "template_score",
        "parser_pred",
        "parser_rule",
        "parser_score",
    ]

    print(
        merged[
            (merged["template_correct"] == False)
            & (merged["parser_correct"] == True)
        ][cols].head(50).to_string(index=False)
    )

    merged.to_csv("outputs/char_ensemble_compare.csv", index=False)
    print("\nsaved outputs/char_ensemble_compare.csv")


if __name__ == "__main__":
    main()
import marimo

__generated_with = "0.11.17"
app = marimo.App(width="full", app_title="TU-Analytics-2025")


@app.cell
def _():
    import polars as pl
    import marimo as mo
    return mo, pl


@app.cell
def _(pl):
    labels_df = (
        pl.read_csv("Labels.csv")
        .filter(~pl.col("VariableName").str.starts_with("State"))
        .with_columns(pl.col("Value").cast(pl.Int64))
    )
    return (labels_df,)


@app.cell
def _(labels_df, pl):
    data_df = pl.read_csv("data.csv")

    mapping_dict = {
        var: dict(zip(sub_df["Value"], sub_df["ValueLabel"]))
        for var, sub_df in labels_df.group_by("VariableName")
    }

    data_df = data_df.with_columns(
        [
            pl.col(col).cast(pl.Utf8).replace(mapping)
            for col, mapping in mapping_dict.items()
        ]
    )
    return data_df, mapping_dict


@app.cell
def _(data_df):
    print(data_df.head())
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ###Grouping Columns
        Based on TU's [strategic plan](https://www.towson.edu/about/mission/strategic-plan/targets-2030.html), create the following strategic categories:

        - Student Success
        - Access & Equity
        - Academic Resources
        - Career & Economic Outcomes
        - Innovation & Research
        - Sustainability & Efficiency
        - Community Engagement
        """
    )
    return


@app.cell
def _():
    def cols_from_range(start: int, end: int, cols: list[str]) -> list[str]:
        return cols[start-1:end]
    return (cols_from_range,)


@app.cell
def _(cols_from_range, data_df):
    groupings = {
        "Student Success": cols_from_range(30, 34, data_df.columns) + [data_df.columns[105]],
        "Access & Equity": cols_from_range(23, 29, data_df.columns) + cols_from_range(86, 97, data_df.columns),
        "Academic Resources": cols_from_range(66, 73, data_df.columns) + [data_df.columns[106]],
        "Career & Economic Outcomes": cols_from_range(18, 22, data_df.columns),
        "Innovation & Research": cols_from_range(35, 65, data_df.columns), 
        "Sustainability & Efficiency": cols_from_range(35, 65, data_df.columns),
        "Community Engagement": []  # TBD or qualitative for now
    }
    return (groupings,)


@app.cell
def _(groupings):
    for group, cols in groupings.items():
        print(f"\n{group} ({len(cols)} columns):")
        for col in cols:
            print("  ", col)
    return col, cols, group


@app.cell
def _(groupings):
    import json
    with open("strategic_groupings.json", "w") as f:
        json.dump(groupings, f)
    return f, json


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""###Weight Scenarios""")
    return


@app.cell
def _():
    # group = "Student Success"
    # cols = groupings[group]
    return


@app.cell
def _(pl):
    def z_score(df: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
        return df.select([
            ((pl.col(col) - df.select(pl.col(col).mean()).item()) / df.select(pl.col(col).mean()).item()).alias(col)
            for col in cols
        ])
    return (z_score,)


@app.cell
def _(cols, data_df, z_score):
    normalized = z_score(data_df, cols)
    return (normalized,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ###Peer Institutions
        When thinking about peer institutions we may consider those with similar

        - Student-to-faculty ratios
        - In person/online distribution
        - Racial distributions
        - Level distributions
        - Sticker price
        - Admissions rate
        - Revenues & Expenditures
        - Socio-Economic Distribution (proxied by fin aid)
        - Region
        - Primary public control

        This may be too restrictive so maybe we can whittle it down. Or create measured rankings for each of these categories and do some least squared error compared to towson. But if one school is identical to towson in all but one category, it might have a similar score to a school that is kind of similar to TU accross all categories. Which is more of a peer?
        """
    )
    return


if __name__ == "__main__":
    app.run()

import marimo

__generated_with = "0.11.17"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Exploring the data

        Ideas:
        - Train a model that eventually matches with real world rankings
        - Determine the features that have the most impact based on the weights
        """
    )
    return


@app.cell
def _():
    import altair as alt
    import polars as pl
    return alt, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Mapping labels with string classification

        Un-doing the categorical encoding already presented to get a better understanding of the data.
        """
    )
    return


@app.cell
def _(pl):
    labels_df = (
        pl.read_csv("Labels.csv")
        .filter(~pl.col("VariableName").str.starts_with("State"))
        .with_columns(pl.col("Value").cast(pl.Int64))
    )
    return (labels_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Data decoding""")
    return


@app.cell
def _(labels_df, pl):
    data_df = pl.read_csv("data.csv").drop(
        "UnitID", "Institution (entity) name (HD2023)"
    )

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


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Add real world rankings

        Add columns on where an institution ranks

        Each real world ranking such as:
        - U.S. News Best Colleges
        - Wall Street Journal
        - Princeton Review
        - Forbes
        - Washington Monthly

        Analyze and look for patterns for each rankings with the variables from the original data
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### External College Rankings

        External rankings use different annotations so an exact merge is not always possible.
        Use fuzzy matching with a threshold of 97 for accuracy

        #### Niche

        TU ranks 341. 

        #### Forbes 2025

        TU ranks 174

        #### Times Higher Ed

        TU ranks 391
        """
    )
    return


@app.cell
def _(pl):
    from json import load

    with open("Ranking_datasets/niche-800.json", "r") as f:
        niche_json = load(f)

    with open("Ranking_datasets/Forbes-Ranking-2025.json") as f:
        forbes_json = load(f)

    with open("Ranking_datasets/timeshighered-2022.json") as f:
        times_json = load(f)

    niche_rankings = {}
    forbes_rankings = {}
    highered_rankings = {}
    for idx, university in enumerate(niche_json["entities"]):
        niche_rankings[university["content"]["entity"]["name"]] = idx + 1

    niche_df = pl.DataFrame(
        {"Institution": niche_rankings.keys(), "Ranking": niche_rankings.values()}
    )

    for university in forbes_json["organizationList"]["organizationsLists"]:
        forbes_rankings[university["organizationName"]] = university["rank"]

    forbes_df = pl.DataFrame(
        {
            "Institution": forbes_rankings.keys(),
            "Ranking": forbes_rankings.values(),
        }
    )
    for university in times_json["data"]:
        highered_rankings[university["name"]] = university["rank_order"]

    highered_df = pl.DataFrame(
        {
            "Institution": highered_rankings.keys(),
            "Ranking": highered_rankings.values(),
        }
    ).with_columns(pl.col("Ranking").cast(pl.Int32))
    return (
        f,
        forbes_df,
        forbes_json,
        forbes_rankings,
        highered_df,
        highered_rankings,
        idx,
        load,
        niche_df,
        niche_json,
        niche_rankings,
        times_json,
        university,
    )


@app.cell
def _(data_df, forbes_df, highered_df, niche_df, pl):
    from thefuzz import fuzz


    def fuzzy_join(
        left_df: pl.DataFrame,
        right_df: pl.DataFrame,
        left_on: str = "Institution Name",
        right_on: str = "Institution",
        threshold: int = 97,
    ) -> pl.DataFrame:
        """
        Perform a fuzzy join between two Polars DataFrames using token_sort_ratio.
        Parameters:
        -----------
        left_df : pl.DataFrame
            The left DataFrame to join
        right_df : pl.DataFrame
            The right DataFrame to join
        left_on : str
            The column name in the left DataFrame to match
        right_on : str
            The column name in the right DataFrame to match
        threshold : int, optional (default=80)
            Minimum similarity score to consider a match (0-100)

        Returns:
        --------
        pl.DataFrame
            Joined DataFrame with matches above the similarity threshold
        """
        # Create cartesian product of DataFrames
        cross_df = left_df.join(right_df, how="cross")

        # Apply fuzzy matching using token_sort_ratio
        matched_df = cross_df.with_columns(
            [
                pl.struct([pl.col(left_on), pl.col(right_on)])
                .map_elements(
                    lambda x: fuzz.token_sort_ratio(
                        str(x[left_on]), str(x[right_on])
                    ),
                    return_dtype=pl.Int64,
                )
                .alias("similarity_score")
            ]
        ).filter(pl.col("similarity_score") >= threshold)

        return matched_df


    # Map the rankings from external sources with the original data
    merged_niche = fuzzy_join(data_df, niche_df)
    merged_forbes = fuzzy_join(data_df, forbes_df)
    merged_highered = fuzzy_join(data_df, highered_df)
    return fuzz, fuzzy_join, merged_forbes, merged_highered, merged_niche


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Financial Aid

        I'm going to guess that TU excels at financial aid

        Find the relationship between student success and the percentage of students awarded grants etc.

        Study shows that students tend to perform best when they're not stressed about paying for college.

        Student retention also tends to be so much better

        TODO: 

        - Rank where TU falls in terms of average amount of grants awarded
        """
    )
    return


@app.cell
def _(alt, col_sel, pl):
    def TU_Compare(df: pl.DataFrame, rank_dif: int, cols):
        """
        Returns a chart to compare TU with institutions based on ranking from the dataframe passed
        """
        sorted_rank = df.sort(by="Ranking").with_row_index()

        TU_index = sorted_rank.filter(
            pl.col("Institution Name") == "Towson University"
        ).item(0, "index")

        sorted_rank = sorted_rank[
            TU_index - rank_dif : TU_index + rank_dif
        ].select("Institution Name", "Ranking", col_sel[cols])

        df_melted = sorted_rank.unpivot(index="Institution Name").filter(
            pl.col("variable") != "Ranking"
        )
        return (
            alt.Chart(df_melted)
            .mark_bar()
            .encode(
                x=alt.X(
                    "Institution Name:N",
                    title="Institution Name",
                    sort=alt.EncodingSortField(field="Ranking", order="ascending"),
                ),
                y=alt.Y("value:Q", title="Value"),
                xOffset="variable:N",
                color=alt.Color("variable:N", title="Category"),
                tooltip=["Institution Name", "variable", "value"],
            )
            .properties(width=1500, height=900)
        )
    return (TU_Compare,)


@app.cell
def _(TU_Compare, merged_forbes):
    TU_Compare(merged_forbes, 20, "Admissions")
    return


@app.cell
def _(merged_forbes, peers, pl):
    merged_forbes.filter(
        pl.col("Institution Name").is_in(peers + ["Towson University"])
    ).select(["Ranking", "Institution Name"])
    return


@app.cell
def _(merged_niche, peers, pl):
    merged_niche.filter(
        pl.col("Institution Name").is_in(peers + ["Towson University"])
    ).select(["Ranking", "Institution Name"])
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        According to https://www.towson.edu/ir/reports.html the following are the peer institutions that TU leadership would like to compete with. 

        For the competition, we have to create our own list of institutions.
        """
    )
    return


@app.cell
def _():
    peers = [
        "Appalachian State University",
        "California State University-Fullerton",
        "Indiana University of Pennsylvania-Main Campus",
        "James Madison University",
        "Minnesota State University-Mankato",
        "Montclair State University",
        "University of Massachusetts-Dartmouth",
        "University of North Carolina at Charlotte",
        "University of North Carolina Wilmington",
        "West Chester University of Pennsylvania",
        "Western Washington University",
    ]
    return (peers,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Column Groupings

        To avoid the possibility of carpal tunnel, we group the columns.
        It's a (-2, +1) column ranges
        """
    )
    return


@app.cell
def _(pl):
    column_ranges = {
        "Degrees Conferred": (16, 21),
        "Financial Aid": (21, 27),
        "fin_perc": (21, 23),
        "fin_avg": (23, 25),
        "Student Success": (28, 32),
        "Revenues": (32, 46),
        "Expenditures": (46, 61),
        "Library": (64, 71),
        "Admissions": (71, 74),
        "Race": (87, 96),
        "Population": (81, 84),
        "Graduation Rate": (29, 32),
    }

    col_sel = {k: pl.nth(range(*v)) for k, v in column_ranges.items()}
    return col_sel, column_ranges


@app.cell
def _(col_sel, data_df):
    data_df.select("Institution Name", col_sel["Admissions"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### Grouping by population size""")
    return


@app.cell
def _(data_df, pl):
    TU_pop = data_df.filter(pl.col("Institution Name") == "Towson University")[
        "Grand total (EF2023A  All students total)"
    ][0]

    institutes_list = data_df.filter(
        pl.col("Grand total (EF2023A  All students total)").is_between(
            TU_pop - 2000, TU_pop + 2000
        )
    )["Institution Name"]
    return TU_pop, institutes_list


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### How TU compares to other Maryland colleges""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Relationship between admission rate and student success

        Compare the ranking of TU with next and previous 5 ranked institutions
        So does TU need to be more selective in order to rank higher? 

        Swear to god some kids here should've stayed in High School
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        Towson's research expenses is only at 2%. 

        Find the relationship between research expenses and the number of international students
        """
    )
    return


@app.cell
def _(col_sel, merged_niche, pl):
    merged_niche.select(
        "Institution Name",
        "Ranking",
        "U.S. Nonresident total (EF2023A  All students total)",
        "Grand total (EF2023  All students total)",
        col_sel["Expenditures"],
    ).with_columns(
        (
            pl.col("U.S. Nonresident total (EF2023A  All students total)")
            / pl.col("Grand total (EF2023  All students total)")
        )
        .alias("International student in %")
        .round(2)
    ).select(
        "Ranking",
        "Research expenses as a percent of total core expenses (GASB) (DRVF2023)",
        "Institution Name",
        "International student in %",
    ).filter(
        ~pl.col(
            "Research expenses as a percent of total core expenses (GASB) (DRVF2023)"
        ).is_null(),
    )
    return


@app.cell
def _(col_sel, data_df):
    data_df.select(col_sel["Financial Aid"]).describe()
    return


@app.cell
def _(data_df, pl):
    f"""TU is slightly above average in terms of federal pell grant awarded to UG students at
    {
        data_df.filter(pl.col("Institution Name") == "Towson University").item(
            0,
            "Average amount Federal Pell grant aid awarded to undergraduate students (SFA2223)",
        )
    }
    """
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Peer Institutions

        Peer institutions will have similar:

        - Carnegie classification
        - Population size
        - Admissions rate
        """
    )
    return


@app.cell
def _(data_df, pl):
    TU_population = 19527
    TU_admission_rate = 0.83

    data_df.with_columns(
        (
            pl.col("Admissions total (ADM2023)")
            / pl.col("Applicants total (ADM2023)")
        ).alias("Admission Rate")
    ).filter(
        (pl.col("Carnegie Classification 2021: Basic (HD2023)")
        == "Master's Colleges & Universities: Larger Programs")
        & (pl.col("Grand total (EF2023  All students total)").is_between(
            TU_population - 8000, TU_population + 8000
        ))
        & (pl.col("Admission Rate").is_between(
            TU_admission_rate - 0.10, TU_admission_rate + 0.10
        ))
    )
    return TU_admission_rate, TU_population


@app.cell
def _(data_df, pl):
    data_df.filter(
        pl.col("Carnegie Classification 2021: Basic (HD2023)")
        == "Master's Colleges & Universities: Larger Programs"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Data Preprocessing""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Weights of each different types of expenses

        Use a learning to rank model to rank each institution

        Get the weights of the greatest accuracy
        """
    )
    return


@app.cell
def _(col_sel, column_ranges, merged_forbes, pl):
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()

    scaled_df = pl.DataFrame(
        scaler.fit_transform(
            merged_forbes.select(col_sel["Expenditures"]).to_numpy()
        ),
        schema=merged_forbes.columns[
            column_ranges["Expenditures"][0] : column_ranges["Expenditures"][1]
        ],
    )
    scaled_df
    return MinMaxScaler, scaled_df, scaler


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

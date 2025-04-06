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
    from pprint import pprint
    return alt, pl, pprint


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


@app.cell
def _(data_df):
    data_df.columns
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Use of real work rankings

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
def _(mo):
    mo.md(
        r"""
        ### Comparison by Financial Aid

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
def _(TU_dict, merged_forbes, pl):
    # Sort by external rankings and include a row index. This will be used as a relative ranking
    sorted_rank = merged_forbes.sort(by="Ranking").with_row_index()

    rank_dif = 30

    # Get the relative ranking of TU
    TU_index = sorted_rank.filter(
        pl.col("Institution Name") == "Towson University"
    ).item(0, "index")


    sorted_rank = sorted_rank.filter(
        pl.col("Carnegie Classification 2021: Basic (HD2023)")
        == TU_dict["Carnegie Classification 2021: Basic (HD2023)"]
    )[TU_index - rank_dif : TU_index + rank_dif]
    return TU_index, rank_dif, sorted_rank


@app.cell
def _(sorted_rank):
    sorted_rank
    return


@app.cell
def _(merged_forbes, peers, pl):
    merged_forbes.filter(
        pl.col("Institution Name").is_in(peers + ["Towson University"])
    ).select(["Ranking", "Institution Name"])
    return


@app.cell
def _(col_sel, data_df):
    data_df.select("Institution Name", col_sel["Other success"])
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
def _(mo):
    mo.md(
        r"""
        ## How TU ranks nationally compares

        We want to assign weights for each ranking
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Affordability and social mobility

        See if TU has good social mobility

        Look at the percent of pell grant undergrads, avg amount of federal pell grant awareded, and cost to go to school 
        """
    )
    return


@app.cell
def _(TU_dict, col_sel, data_df, pl):
    no_brokeAss_colleges = data_df.filter(
        pl.col("Carnegie Classification 2021: Undergraduate Profile (HD2023)")
        == TU_dict["Carnegie Classification 2021: Undergraduate Profile (HD2023)"]
    )

    no_brokeAss_colleges.select(col_sel["Financial Aid"]).describe()
    return (no_brokeAss_colleges,)


@app.cell
def _():
    return


@app.cell
def _(mo):
    mo.md(r"""### Diversity and Equity""")
    return


@app.cell
def _():
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Graduate Experience

        - Many graduate students depend on stipend and assistantship to continue their graudate studies
        - Stipend and assistantship comes from research expense but funding could come from other sources
        """
    )
    return


@app.cell
def _(TU_dict):
    f"TU invests {TU_dict['Research expenses as a percent of total core expenses (GASB) (DRVF2023)']}% of their expenses on research"
    return


@app.cell
def _(mo):
    mo.md(r"""### Retention Rate""")
    return


@app.cell
def _(col_sel, data_df):
    data_df.select(col_sel["Other success"]).describe()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Our list of peer Institutions

        Peer institutions will have similar characteristics:

        - Carnegie classification, undergraduate
        - Population size
            - $\pm$ 7000
        - Admissions rate
            - $\pm$ 10%
        """
    )
    return


@app.cell
def _(TU_Compare, TU_dict, col_sel, data_df, pl):
    TU_population = TU_dict["Grand total (EF2023A  All students total)"]
    TU_admission_rate = (
        TU_dict["Admissions total (ADM2023)"]
        / TU_dict["Applicants total (ADM2023)"]
    )
    tolerance_range = {"population": 7000, "admission_rate": 0.08}

    TU_peers = (
        data_df.with_columns(
            (
                pl.col("Admissions total (ADM2023)")
                / pl.col("Applicants total (ADM2023)")
            ).alias("Admission Rate")
        )
        .filter(
            (
                pl.col(
                    "Carnegie Classification 2021: Undergraduate Profile (HD2023)"
                )
                == TU_dict[
                    "Carnegie Classification 2021: Undergraduate Profile (HD2023)"
                ]
            )
            & (
                pl.col("Grand total (EF2023  All students total)").is_between(
                    TU_population - tolerance_range["population"],
                    TU_population + tolerance_range["population"],
                )
            )
            & (
                pl.col("Admission Rate").is_between(
                    TU_admission_rate - tolerance_range["admission_rate"],
                    TU_admission_rate + tolerance_range["admission_rate"],
                )
            )
        )
        .select("Institution Name", col_sel["fin_perc"])
    )
    print(TU_peers.select("Institution Name"))
    # TU_peers
    TU_Compare(TU_peers)
    return TU_admission_rate, TU_peers, TU_population, tolerance_range


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Learning To Rank""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Data pre-processing

        Standardize data
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
def _(data_df, merged_forbes, pl):
    import polars.selectors as cs
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier()
    train_data_df = data_df.select(cs.numeric()).filter(pl.all_horizontal(pl.all().is_nan()))

    train_forbes = merged_forbes.select(cs.numeric()).filter(
            ~pl.all_horizontal(pl.all().is_nan()))

    # Sort institution name since it matches by position
    # model.fit(
    #     train_data_df, train_forbes
    # )

    # model.feature_importances_
    return RandomForestClassifier, cs, model, train_data_df, train_forbes


@app.cell
def _(train_data_df):
    train_data_df
    return


@app.cell
def _(cs, data_df):
    data_df.select(cs.numeric())
    return


@app.cell
def _(cs, merged_forbes):
    merged_forbes.select(cs.numeric())
    return


@app.cell
def _(mo):
    mo.md(r"""## Functions""")
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""### Fuzzy Merge""")
    return


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
    mo.md(r"""###Averaging out""")
    return


@app.cell
def _(merged_forbes, merged_highered, merged_niche):
    merged_highered.select("Ranking", "Institution Name").rename(
        {"Ranking": "HigherEd Ranking"}
    ).join(
        merged_forbes.select("Ranking", "Institution Name").rename(
            {"Ranking": "Forbes Ranking"}
        ),
        on="Institution Name",
    ).join(
        merged_niche.select("Ranking", "Institution Name").rename(
            {"Ranking": "HigherEd Ranking"}
        ),
        on="Institution Name",
    )
    return


@app.cell
def _(merged_niche):
    merged_niche
    return


@app.cell
def _(merged_forbes):
    merged_forbes.select("Ranking", "Institution Name")
    return


@app.cell
def _(merged_niche):
    merged_niche.select("Ranking", "Institution Name")
    return


@app.cell
def _(mo):
    mo.md(r"""### Bar chart generator""")
    return


@app.cell
def _(alt, pl):
    def TU_Compare(df: pl.DataFrame):
        """
        Returns a chart to compare TU with a list of institutions
        """

        df_melted = df.unpivot(index="Institution Name").filter(
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
def _(mo):
    mo.md(r"""## Variables""")
    return


@app.cell
def _(mo):
    mo.md(r"""### Column groups""")
    return


@app.cell
def _(col_sel, data_df):
    data_df.select("Institution Name", col_sel["Degrees Conferred"])
    return


@app.cell
def _(pl):
    column_ranges = {
        "Degrees Conferred": (15, 20),
        "Financial Aid": (21, 27),
        "fin_perc": (20, 22),
        "fin_avg": (22, 24),
        "Student Success": (27, 32),
        "Revenues": (32, 46),
        "Expenditures": (46, 61),
        "Library": (64, 71),
        "Admissions": (71, 74),
        "Race": (87, 96),
        "Population": (81, 84),
        "Graduation Rate": (29, 32),
        "Other success": (103, 105),
    }

    col_sel = {k: pl.nth(range(*v)) for k, v in column_ranges.items()}
    return col_sel, column_ranges


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Real world peers

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


@app.cell
def _(TU_Compare, col_sel, data_df, peers, pl):
    TU_Compare(
        data_df.filter(
            pl.col("Institution Name").is_in(peers + ["Towson University"])
        ).select(col_sel["Graduation Rate"], "Institution Name")
    )
    return


@app.cell
def _(mo):
    mo.md(f"""
    ### TU values
    """)
    return


@app.cell
def _(data_df, pl):
    TU_dict = dict(
        zip(
            data_df.filter(
                pl.col("Institution Name") == "Towson University"
            ).columns,
            data_df.filter(pl.col("Institution Name") == "Towson University").row(
                0
            ),
        )
    )
    return (TU_dict,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

import marimo

__generated_with = "0.11.17"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import altair as alt
    import polars as pl
    import polars.selectors as cs
    from sklearn.cluster import KMeans
    return KMeans, alt, cs, mo, pl


@app.cell
def _(KMeans, alt, cs, pl):
    def drop_majority_nulls(df):
        # Total number of rows in your DataFrame
        row_count = df.height

        # Get null count for each column
        null_counts = df.select(
            [pl.col(col).is_null().sum().alias(col) for col in df.columns]
        )

        majority_null_cols = null_counts.unpivot(
            variable_name="column", value_name="null_count"
        ).filter(pl.col("null_count") > (row_count / 1.7))

        to_drop = majority_null_cols["column"].to_list()

        return df.drop(to_drop)


    def get_standardScale_df(df: pl.DataFrame):
        # Normalize data using z-score
        scaler = StandardScaler()
        features = df.select(cs.numeric())
        scaled_features = scaler.fit_transform(features.to_numpy())
        return pl.DataFrame(scaled_features, schema=features.columns).with_columns(df.select("Institution Name"))


    def get_skewed_cols(df):
        numeric_cols = df.select(cs.numeric()).columns
        # Compute skewness for each numeric column
        skew_df = df.select(
            [pl.col(col).skew().alias(col) for col in numeric_cols]
        )
        skew_long = skew_df.unpivot(variable_name="column", value_name="skewness")
        high_skew = skew_long.filter(
            (pl.col("skewness") > 1.0) | (pl.col("skewness") < -1.0)
        )
        return high_skew


    def fill_null_numeric_cols(df):
        features = data_df.select(cs.numeric())
        # Impute missing values with the mean of the column
        for col in features.columns:
            features = features.with_columns(
                pl.col(col).fill_null(features[col].mean()).alias(col)
            )

        return features


    def log_transform(df):
        # Log transformation to alleviate skewed data
        df = df.with_columns(
            [
                pl.when(pl.col(col) <= 0)
                .then(1e-10)
                .otherwise(pl.col(col))
                .log1p()
                .alias(f"{col}")
                for col in get_skewed_cols(df)["column"]
            ]
        )
        return df


    def pre_process(df: pl.DataFrame):
        df = get_standardScale_df(df)

        return df


    labels_df = (
        pl.read_csv("Labels.csv")
        .filter(~pl.col("VariableName").str.starts_with("State"))
        .with_columns(pl.col("Value").cast(pl.Int64))
    )

    data_df = (
        pl.read_csv("data.csv")
        .drop("UnitID", "Institution (entity) name (HD2023)")
        # Filter out non 4-year schools
        .filter(
            pl.col("Carnegie Classification 2021: Undergraduate Profile (HD2023)")
            > 3
        )
    )

    mapping_dict = {
        var: dict(zip(sub_df["Value"], sub_df["ValueLabel"]))
        for var, sub_df in labels_df.group_by("VariableName")
    }


    # Decode categorical columns
    data_df = data_df.with_columns(
        [
            pl.col(col).cast(pl.Utf8).replace(mapping)
            for col, mapping in mapping_dict.items()
        ]
    )

    # data_df = data_df.with_columns(
    #       (
    #         pl.col('Admissions total (ADM2023)') / pl.col("Applicants total (ADM2023)")
    #     ).alias("admission rate"),
    # )

    # Remove columns that have 1/2 of its data missing
    data_df = drop_majority_nulls(data_df)

    data_df_numeric_cols = data_df.select(cs.numeric()).columns

    # Fill in null values in columns with their respective medians
    data_df = data_df.with_columns(
        pl.col(data_df_numeric_cols).fill_null(
            pl.col(data_df_numeric_cols).median()
        )
    )

    # Log transformation to alleviate skewed data
    data_df = data_df.with_columns(
        [
            pl.when(pl.col(col) <= 0)
            .then(1e-10)
            .otherwise(pl.col(col))
            .log1p()
            .alias(f"{col}")
            for col in get_skewed_cols(data_df)["column"]
        ]
    )


    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    cluster_range = (1, 30)
    n_clusters = 8

    # Retrieve numeric columns
    features = data_df.select(cs.numeric())
    # Impute missing values with the mean of the column
    for col in features.columns:
        features = features.with_columns(
            pl.col(col).fill_null(features[col].mean()).alias(col)
        )

    # Normalize data using z-score
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features.to_numpy())
    scaled_df = pl.DataFrame(scaled_features, schema=features.columns)

    # Principal component analysis for clustering
    pca = PCA(n_components=2)
    pca_res = pca.fit_transform(scaled_features)
    pca_df = pl.DataFrame(pca_res, schema=[f"PC{i}" for i in range(1, 3)])
    pca_df = pca_df.with_columns(data_df.select("Institution Name"))

    # Apply K-Means cluster
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kdata = data_df.with_columns(
        pl.Series(name="Cluster", values=kmeans.fit_predict(pca_res))
    )

    # Add cluster group for each Row
    pca_df = pca_df.join(
        kdata.select(["Institution Name", "Cluster"]), on="Institution Name"
    ).with_columns(
        # Track Towson
        (pl.col("Institution Name") == "Towson University").alias("IsTowson")
    )

    towson_x = pca_df.filter(pl.col("IsTowson")).select("PC1").item()
    towson_y = pca_df.filter(pl.col("IsTowson")).select("PC2").item()

    hline = (
        alt.Chart(pl.DataFrame({"y": [towson_y]}))
        .mark_rule(color="red")
        .encode(y="y")
    )
    vline = (
        alt.Chart(pl.DataFrame({"x": [towson_x]}))
        .mark_rule(color="red")
        .encode(x="x")
    )

    b_chart = (
        alt.Chart(pca_df.to_pandas())
        .mark_circle()
        .encode(
            x="PC1",
            y="PC2",
            color=alt.Color(
                "Cluster:N",
                legend=alt.Legend(title="Cluster"),
            ),
            tooltip=["Institution Name", "Cluster"],
        )
        .properties(title="Institution Clusters", width=600, height=400)
    )

    b_chart + hline + vline
    return (
        PCA,
        StandardScaler,
        b_chart,
        cluster_range,
        col,
        data_df,
        data_df_numeric_cols,
        drop_majority_nulls,
        features,
        fill_null_numeric_cols,
        get_skewed_cols,
        get_standardScale_df,
        hline,
        kdata,
        kmeans,
        labels_df,
        log_transform,
        mapping_dict,
        n_clusters,
        pca,
        pca_df,
        pca_res,
        pre_process,
        scaled_df,
        scaled_features,
        scaler,
        towson_x,
        towson_y,
        vline,
    )


@app.cell
def _(KMeans, alt, cluster_range, pl, scaled_df):
    def visualize_elbow():
        inertia = []

        for n in range(*cluster_range):
            kmeans = KMeans(n_clusters=n, random_state=42)
            kmeans.fit(scaled_df)
            inertia.append(kmeans.inertia_)

        # Creating the elbow plot
        elbow_chart = (
            alt.Chart(
                pl.DataFrame(
                    {"clusters": range(*cluster_range), "inertia": inertia}
                )
            )
            .mark_line(point=True)
            .encode(
                x=alt.X("clusters:Q", title="Number of Clusters"),
                y=alt.Y("inertia:Q", title="Inertia"),
            )
            .properties(
                title="Elbow Method for Optimal Number of Clusters",
                width=800,
                height=400,
            )
            .configure_axis(grid=True)
        )

        return elbow_chart


    visualize_elbow()
    return (visualize_elbow,)


@app.cell
def _(data_df, kdata, pca_df, pl):
    TU_cluster = kdata.filter(pl.col("Institution Name") == "Towson University")[
        "Cluster"
    ].item()
    peer_cluster_df = data_df.filter(
        pl.col("Institution Name").is_in(
            pca_df.filter(pl.col("Cluster") == TU_cluster)["Institution Name"]
        )
    )

    tolerance_range = {
        "admission_rate": 0.10,
        "pell grant perc": 5,
    }

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

    filters = [
        (
            pl.col("Carnegie Classification 2021: Undergraduate Profile (HD2023)")
            == TU_dict[
                "Carnegie Classification 2021: Undergraduate Profile (HD2023)"
            ]
        ),
        (
            pl.col(
                "Percent of undergraduate students awarded Federal Pell grants (SFA2223)"
            ).is_between(
                TU_dict[
                    "Percent of undergraduate students awarded Federal Pell grants (SFA2223)"
                ]
                - tolerance_range["pell grant perc"],
                TU_dict[
                    "Percent of undergraduate students awarded Federal Pell grants (SFA2223)"
                ]
                + tolerance_range["pell grant perc"],
            )
        ),
    ]

    peer_institutions = peer_cluster_df.filter(
        pl.reduce(lambda a, b: a & b, filters)
    )["Institution Name"]
    return (
        TU_cluster,
        TU_dict,
        filters,
        peer_cluster_df,
        peer_institutions,
        tolerance_range,
    )


@app.cell
def _(mo):
    mo.md(
        r"""

        Peer Institutions publications count

        To get a local PCA score in research with the number of publications added
        Central Michigan University = 316
        TU = 656

        """
    )
    return


@app.cell
def _(peer_institutions):
    peer_institutions
    return


@app.cell
def _(PCA, data_df, pl):
    def get_categories():
        category_map = {
            "Student Success": [
                "Grand total (C2023_A  First major  Grand total  Bachelor's degree)",
                "Grand total (C2023_A  First major  Grand total  Master's degree)",
                "Grand total (C2023_A  First major  Grand total  Doctor's degree - research/scholarship )",
                "Graduation rate - Bachelor degree within 4 years  total (DRVGR2023)",
                "Graduation rate - Bachelor degree within 5 years  total (DRVGR2023)",
                "Graduation rate - Bachelor degree within 6 years  total (DRVGR2023)",
                "Transfer-out rate - Bachelor cohort (DRVGR2023)",
                "Pell Grant recipients - Bachelor's degree rate within 6 years (DRVGR2023)",
                "Academic support expenses as a percent of total core expenses (GASB) (DRVF2023)",
                "Student service expenses as a percent of total core expenses (GASB) (DRVF2023)",
                "Academic support expenses per FTE (GASB) (DRVF2023)",
                "Student service expenses per FTE (GASB) (DRVF2023)",
                "Full-time retention rate  2023 (EF2023D)",
                "Student-to-faculty ratio (EF2023D)",
                # "Price-to-Earnings Premium for the Median Student",
                # "Median Earnings 10 Years After Initial Enrollment for All Students"
            ],
            "Equity": [
                # "Grand total (EF2023B  Undergraduate  Age under 25 total)",
                # "Grand total (EF2023B  Undergraduate  All age categories total)",
                # "Grand total (EF2023B  Undergraduate  Age 25 and over total)",
                # "Grand total (EF2023  All students total)",
                # "Grand total (EF2023  All students  Undergraduate total)",
                # "Grand total (EF2023  All students  Graduate and First professional)",
                "Grand total men (EF2023A  All students total)",
                "Grand total women (EF2023A  All students total)",
                "American Indian or Alaska Native total (EF2023A  All students total)",
                "Asian total (EF2023A  All students total)",
                "Black or African American total (EF2023A  All students total)",
                "Hispanic total (EF2023A  All students total)",
                "Native Hawaiian or Other Pacific Islander total (EF2023A  All students total)",
                "White total (EF2023A  All students total)",
                "Two or more races total (EF2023A  All students total)",
                "Race/ethnicity unknown total (EF2023A  All students total)",
                "U.S. Nonresident total (EF2023A  All students total)",
            ],
            "Access": [
                "Percent of undergraduate students awarded federal  state  local  institutional or other sources of grant aid (SFA2223)",
                "Percent of undergraduate students awarded Federal Pell grants (SFA2223)",
                "Average amount Federal Pell grant aid awarded to undergraduate students (SFA2223)",
                "Average amount of federal  state  local  institutional or other sources of grant aid awarded to undergraduate students (SFA2223)",
                "Percent of undergraduate students awarded federal student loans (SFA2223)",
                "Average amount of federal student loans awarded to undergraduate students (SFA2223)",
                "Average net price-students awarded grant or scholarship aid  2022-23 (SFA2223)",
                "Tuition and fees as a percent of core revenues (GASB) (DRVF2023)",
                "State appropriations as percent of core revenues  (GASB) (DRVF2023)",
                "Local appropriations as a percent of core revenues (GASB) (DRVF2023)",
                "Revenues from tuition and fees per FTE (GASB) (DRVF2023)",
                "Revenues from state appropriations per FTE (GASB) (DRVF2023)",
                "Revenues from local appropriations per FTE (GASB) (DRVF2023)",
                "Total price for in-state students living on campus 2023-24 (DRVIC2023)",
                "Total price for out-of-state students living on campus 2023-24 (DRVIC2023)",
                "Private gifts  grants  and contracts as a percent of core revenues (GASB) (DRVF2023)",
                "Revenues from private gifts  grants  and contracts per FTE (GASB) (DRVF2023)",
                # "admission rate",
                # "EMI Score (low-income percentile rank*percentage pell)",
            ],
            "Academic Resources": [
                "Instruction expenses as a percent of total core expenses (GASB) (DRVF2023)",
                "Instruction expenses per FTE  (GASB) (DRVF2023)",
                "Total library expenditures per FTE (DRVAL2023)",
                "Students enrolled exclusively in distance education courses (EF2023A_DIST  Undergraduate total)",
                "Students enrolled in some but not all distance education courses (EF2023A_DIST  Undergraduate total)",
                "Student not enrolled in any distance education courses (EF2023A_DIST  Undergraduate total)",
                "Total library FTE staff (AL2023)",
                "Total physical library circulations (books and media) (AL2023)",
                "Total library circulations (physical and digital/electronic) (AL2023)",
                "Total digital/electronic circulations (books and media) (AL2023)",
                "Student-to-faculty ratio (EF2023D)",
            ],
            "Innovation & Research": [
                "Government grants and contracts as a percent of core revenues (GASB) (DRVF2023)",
                "Private gifts  grants  and contracts as a percent of core revenues (GASB) (DRVF2023)",
                "Revenues from government grants and contracts per FTE (GASB) (DRVF2023)",
                "Revenues from private gifts  grants  and contracts per FTE (GASB) (DRVF2023)",
                "Research expenses as a percent of total core expenses (GASB) (DRVF2023)",
                "Research expenses per FTE  (GASB) (DRVF2023)",
                "Physical books as a percent of the total library collection (DRVAL2023)",
                "Physical media as a percent of the total library collection (DRVAL2023)",
                "Physical serials as a percent of the total library collection (DRVAL2023)",
                "Digital/Electronic books as a percent of the total library collection (DRVAL2023)",
                "Databases as a percent of the total library collection (DRVAL2023)",
                "Digital/Electronic media as a percent of the total library collection (DRVAL2023)",
                "Digital/Electronic serials as a percent of the total library collection (DRVAL2023)",
                "Grand total (C2023_A  First major  Grand total  Doctor's degree - research/scholarship )",
            ],
            "Sustainability & Efficiency": [
                "Investment return as a percent of core revenues (GASB) (DRVF2023)",
                "Other revenues as a percent of core revenues (GASB) (DRVF2023)",
                "Revenues from investment return per FTE (GASB) (DRVF2023)",
                "Other core revenues per FTE (GASB) (DRVF2023)",
                "Institutional support expenses as a percent of total core expenses (GASB) (DRVF2023)",
                "Other core expenses as a percent of total core expenses (GASB) (DRVF2023)",
                "Institutional support expenses per FTE (GASB) (DRVF2023)",
                "All other core expenses per FTE (GASB) (DRVF2023)",
                "Endowment assets (year end) per FTE enrollment (GASB) (DRVF2023)",
                "Equity ratio (GASB) (DRVF2023)",
            ],
            "Community Engagement": [
                "Public service expenses as a percent of total core expenses (GASB) (DRVF2023)",
                "Public service expenses per FTE (GASB) (DRVF2023)"
            ],
        }

        return category_map


    def get_categories_no_expense(category_map):
        filtered_map = {}

        for category, columns in category_map.items():
            filtered_columns = [
                col
                for col in columns
                if "expenditure" not in col.lower()
                and "expenses" not in col.lower()
            ]
            filtered_map[category] = filtered_columns

        return filtered_map


    def get_category_scores(df: pl.DataFrame, categories: dict):
        """
        Calculate the composite score of a category of columns
        """
        map = {}
        for key, group in categories.items():
            X = df.select(group).to_numpy()
            map[key] = PCA(n_components=1).fit_transform(X)[:, 0]

        return pl.DataFrame(map).with_columns(data_df.select("Institution Name"))


    def get_total_scores(df: pl.DataFrame) -> pl.DataFrame:
        """
        Multiply the composite score of each category and sum them to get an overall score
        """
        weights = {
            "Student Success": 0.37,
            "Access": 0.09,
            "Equity": 0.09,
            "Academic Resources": 0.15,
            "Innovation & Research": 0.12,
            "Sustainability & Efficiency": 0.06,
            "Community Engagement": 0.17,
        }

        weighted_expr = sum(
            pl.col(cat) * weight
            for cat, weight in weights.items()
            if cat in df.columns
        )
        return df.with_columns(weighted_expr.alias("Total Score"))
    return (
        get_categories,
        get_categories_no_expense,
        get_category_scores,
        get_total_scores,
    )


@app.cell
def _():
    return


@app.cell
def _(get_categories, get_category_scores, get_total_scores, scaled_df):
    total_ranking_df = (
        get_total_scores(get_category_scores(scaled_df, get_categories()))
        .sort(by="Total Score", descending=True)
        .with_row_index(offset=1)
    )

    total_ranking_df
    return (total_ranking_df,)


@app.cell
def _(get_categories, mo):
    category_choose = mo.ui.dropdown(
        options=get_categories().keys()
    )
    category_choose
    return (category_choose,)


@app.cell
def _(category_choose, get_categories, get_category_scores, scaled_df):
    # National ranking
    get_category_scores(scaled_df, get_categories()).sort(by=category_choose.value, descending=True).with_row_index(offset=1)
    return


@app.cell
def _(category_choose, get_categories, get_category_scores, pl, scaled_df):
    get_category_scores(scaled_df, get_categories()).sort(by=category_choose.value, descending=True).with_row_index(offset=1).filter(pl.col("Institution Name") == "Towson University")
    return


@app.cell
def _(
    category_choose,
    get_categories,
    get_category_scores,
    peer_institutions,
    pl,
    scaled_df,
):
    # Ranking by peers
    get_category_scores(scaled_df, get_categories()).filter(
        pl.col("Institution Name").is_in(peer_institutions)
    ).sort(by=category_choose.value, descending=True).with_row_index(offset=1)
    return


@app.cell
def _(category_choose, data_df, get_categories):
    data_df.select(get_categories()[category_choose.value]).describe()
    return


@app.cell
def _(category_choose, data_df, get_categories, pl):
    data_df.select(get_categories()[category_choose.value] + ["Institution Name"]).filter(pl.col("Institution Name") == "Towson University")
    return


@app.cell
def _(get_categories, get_categories_no_expense):
    get_categories_no_expense(get_categories())
    return


@app.cell
def _(PCA, get_categories, get_categories_no_expense, pl, scaled_df):
    import numpy as np


    def compare_pca(
        with_expense: dict, wo_expense: dict, df: pl.DataFrame, category: str
    ):
        pca_with = PCA(n_components=1).fit_transform(
            df.select(with_expense[category]).to_numpy()
        )
        pca_without = PCA(n_components=1).fit_transform(
            df.select(wo_expense[category]).to_numpy()
        )
        corrcoef = np.corrcoef(pca_with[:, 0], pca_without[:, 0])[0, 1]
        return (
            f"expense has no impact with a correlation coefficient of {corrcoef}"
            if corrcoef >= 0.5
            else f"expense has impact with a correlation coefficient of {corrcoef}"
        )


    a = get_categories()
    # a.pop("Community Engagement")
    for keyy in get_categories().keys():
        print(
            f"{compare_pca(a, get_categories_no_expense(a), scaled_df, keyy)} on {keyy}"
        )
    return a, compare_pca, keyy, np


@app.cell
def _(PCA, alt, get_categories, pl, scaled_df):
    import matplotlib.pyplot as plt
    import seaborn as sns


    def plot_pca_loadings(
        df: pl.DataFrame, group_name: str, feature_names: list[str]
    ) -> alt.Chart:
        X = df.select(feature_names).to_numpy()
        pca = PCA(n_components=1)
        pca.fit(X)

        loadings = pca.components_[0]
        loading_df = pl.DataFrame({"Feature": feature_names, "Loading": loadings})

        # Build Altair chart
        chart = (
            alt.Chart(loading_df)
            .mark_bar()
            .encode(
                x=alt.X("Loading:Q", title="PC1 Loading"),
                y=alt.Y("Feature:N", sort="-x", title=None),
                color=alt.condition(
                    "datum.Loading > 0", alt.value("#3B82F6"), alt.value("#EF4444")
                ),
                tooltip=["Feature", "Loading"],
            )
            .properties(
                title=f"PCA Loadings for {group_name}", width=600, height=400
            )
        )
        return chart


    for k in get_categories().keys():
        plot_pca_loadings(scaled_df, k, get_categories()[k]).save(
            f"{k}.png", scale_factor=2
        )

    # for k in get_categories_without_expense().keys():
    #     plot_pca_loadings(scaled_df, k, get_categories_without_expense()[k]).save(
    #         f"{k}_no_expense.png", scale_factor=2
    #     )
    return k, plot_pca_loadings, plt, sns


@app.cell
def _(pl):
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


    def add_race_ratio(df: pl.DataFrame):
        cols = [
            (
                pl.col(
                    "American Indian or Alaska Native total (EF2023A  All students total)"
                )
                / pl.col("Grand total (EF2023A  All students total)")
            ).alias("indian ratio"),
            (
                pl.col("Asian total (EF2023A  All students total)")
                / pl.col("Grand total (EF2023A  All students total)")
            ).alias("asian ratio"),
            (
                pl.col(
                    "Black or African American total (EF2023A  All students total)"
                )
                / pl.col("Grand total (EF2023A  All students total)")
            ).alias("black ratio"),
            (
                pl.col("Hispanic total (EF2023A  All students total)")
                / pl.col("Grand total (EF2023A  All students total)")
            ).alias("latin ratio"),
            (
                pl.col(
                    "Native Hawaiian or Other Pacific Islander total (EF2023A  All students total)"
                )
                / pl.col("Grand total (EF2023A  All students total)")
            ).alias("island ratio"),
            (
                pl.col("White total (EF2023A  All students total)")
                / pl.col("Grand total (EF2023A  All students total)")
            ).alias("white ratio"),
            (
                pl.col("Two or more races total (EF2023A  All students total)")
                / pl.col("Grand total (EF2023A  All students total)")
            ).alias("tworace ratio"),
            (
                pl.col(
                    "Race/ethnicity unknown total (EF2023A  All students total)"
                )
                / pl.col("Grand total (EF2023A  All students total)")
            ).alias("unknownrace ratio"),
            (
                pl.col("U.S. Nonresident total (EF2023A  All students total)")
                / pl.col("Grand total (EF2023A  All students total)")
            ).alias("nonresident ratio"),
        ]

        return df.with_columns(cols)


    def add_thirdway(df: pl.DataFrame):
        premium_df = pl.read_csv(
            "Third-Way-Price-to-Earnings-Premiums-2024.csv",
            columns=[
                "Institution Name",
                "Median Earnings 10 Years After Initial Enrollment for All Students",
                "Price-to-Earnings Premium for the Median Student",
            ],
        )

        mobility_df = pl.read_csv(
            "Third-Way-Economic-Mobility-Index-2024.csv",
            columns=[
                "Institution Name",
                "EMI Score (low-income percentile rank*percentage pell)",
            ],
        )

        return fuzzy_join(
            premium_df.join(mobility_df, on="Institution Name").rename(
                {"Institution Name": "Institution"}
            ),
            df,
        ).drop("Institution")
    return add_race_ratio, add_thirdway, fuzz, fuzzy_join


@app.cell
def _(
    add_race_ratio,
    data_df,
    drop_majority_nulls,
    get_categories,
    get_category_scores,
    get_standardScale_df,
):
    foo = data_df
    foo = drop_majority_nulls(foo)
    foo = add_race_ratio(foo)
    foo = get_standardScale_df(foo)
    # foo = add_thirdway(foo)

    get_category_scores(foo, get_categories()).sort(by="Student Success")

    # foo = get_standardScale_df(foo)
    return (foo,)


@app.cell
def _():
    # def cock():
    #     for k in get_categories().keys():
    #         plot_pca_loadings(foo, k, get_categories()[k]).save(
    #             f"{k}_with_midway.png", scale_factor=2
    #         )
    # cock()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

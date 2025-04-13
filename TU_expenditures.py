import marimo

__generated_with = "0.11.17"
app = marimo.App(width="full")


@app.cell
def _():
    import altair as alt
    import polars as pl
    import polars.selectors as cs
    from sklearn.cluster import KMeans
    return KMeans, alt, cs, pl


@app.cell
def _(cs, pl):
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


    def get_skewed_cols(df):
        numeric_cols = df.select(cs.numeric()).columns
        # Compute skewness for each numeric column
        skew_df = df.select(
            [pl.col(col).skew().alias(col) for col in numeric_cols]
        )
        print(skew_df)
        skew_long = skew_df.unpivot(variable_name="column", value_name="skewness")
        high_skew = skew_long.filter(
            (pl.col("skewness") > 1.0) | (pl.col("skewness") < -1.0)
        )
        return high_skew
    return drop_majority_nulls, get_skewed_cols


@app.cell
def _(cs, drop_majority_nulls, get_skewed_cols, pl):
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

    data_df = drop_majority_nulls(data_df)

    data_df_numeric_cols = data_df.select(cs.numeric()).columns

    data_df = data_df.with_columns(
        pl.col(data_df_numeric_cols).fill_null(
            pl.col(data_df_numeric_cols).median()
        )
    )

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

    data_df
    return data_df, data_df_numeric_cols, labels_df, mapping_dict


@app.cell
def _():
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Peer Institutions

        To get our list of peer institutions
        """
    )
    return


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
def _(StandardScaler, cs, data_df, features, pl):
    def get_standardScale_df(df):
        # Normalize data using z-score
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features.to_numpy())
        return pl.DataFrame(scaled_features, schema=features.columns)


    def fill_null_numeric_cols(df):
        features = data_df.select(cs.numeric())
        # Impute missing values with the mean of the column
        for col in features.columns:
            features = features.with_columns(
                pl.col(col).fill_null(features[col].mean()).alias(col)
            )

        return features
    return fill_null_numeric_cols, get_standardScale_df


@app.cell
def _(features):
    import numpy as np

    np.nanmin(features.to_numpy())
    return (np,)


@app.cell
def _(KMeans, alt, cs, data_df, pl):
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    cluster_range = (1, 30)
    n_clusters = 8
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

    # Principal component analysis
    pca = PCA(n_components=2)
    pca_res = pca.fit_transform(scaled_features)
    pca_df = pl.DataFrame(pca_res, schema=[f"PC{i}" for i in range(1, 3)])
    pca_df = pca_df.with_columns(data_df.select("Institution Name"))

    # Apply K-Mean
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kdata = data_df.with_columns(
        pl.Series(name="Cluster", values=kmeans.fit_predict(pca_res))
    )


    pca_df = pca_df.join(
        kdata.select(["Institution Name", "Cluster"]), on="Institution Name"
    ).with_columns(
        (pl.col("Institution Name") == "Towson University").alias("IsTowson")
    )

    towson_x = pca_df.filter(pl.col("IsTowson")).select("PC1").item()
    towson_y = pca_df.filter(pl.col("IsTowson")).select("PC2").item()

    hline = alt.Chart(pl.DataFrame({"y": [towson_y]})).mark_rule(color='red').encode(y="y")
    vline = alt.Chart(pl.DataFrame({"x": [towson_x]})).mark_rule(color='red').encode(x="x")

    # 6. Visualization
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
        features,
        hline,
        kdata,
        kmeans,
        n_clusters,
        pca,
        pca_df,
        pca_res,
        scaled_df,
        scaled_features,
        scaler,
        towson_x,
        towson_y,
        vline,
    )


@app.cell
def _(kmeans, np, scaled_df):
    def test():
        sig_features = []
        X = scaled_df.to_numpy()
        cluster_labels = kmeans.labels_
        [print(type(i)) for i in np.unique(cluster_labels)]
    test()
    
    return (test,)


@app.cell
def _(kmeans, np, scaled_df):
    import scipy.stats as stats
    import pandas as pd

    def anova():
        sig_features = []
        X = scaled_df.to_pandas()
        cluster_labels = kmeans.labels_
        for col in X.columns:
            groups = [X[col][cluster_labels == i] for i in np.unique(cluster_labels)]
    
            # Perform ANOVA
            f_stat, p_value = stats.f_oneway(*groups)
        
            if p_value < 0.05:
                sig_features.append((col, p_value))

        print("Significant features across clusters:")
        return pd.DataFrame(sig_features, columns=["Feature", "p-value"])

    anova()
    return anova, pd, stats


@app.cell
def _(alt, kdata, pl, scaled_df):
    def analyze_clusters():
        cluster_means = kdata.group_by("Cluster").agg([
            pl.mean(col).alias(col) for col in scaled_df.columns
        ])
        return cluster_means

    def plot_clusters(df: pl.DataFrame):
        cluster_means = df.group_by("Cluster").mean()
        cluster_means_long = cluster_means.unpivot(variable_name='Feature', value_name='Mean')
        # Create a bar chart with Altair
        chart = alt.Chart(cluster_means_long).mark_bar().encode(
            x=alt.X('Feature:N', title='Feature'),
            y=alt.Y('Mean Value:Q', title='Mean Value'),
            color=alt.Color('Cluster:N', title='Cluster'),
            column=alt.Column('Cluster:N', title='Cluster')
        ).properties(
            width=200,  # Width of each individual chart
            height=300  # Height of each individual chart
        )
        return chart

    # plot_clusters(kdata)
    kdata.group_by("Cluster").mean()
    return analyze_clusters, plot_clusters


@app.cell
def _(pl):
    col_mean = [
        pl.mean(
            "Total price for in-state students living on campus 2023-24 (DRVIC2023)"
        ),
        pl.mean(
            "Total price for out-of-state students living on campus 2023-24 (DRVIC2023)"
        ),
        pl.mean(
            "Average amount Federal Pell grant aid awarded to undergraduate students (SFA2223)"
        ),
        pl.mean(
            "Average net price-students awarded grant or scholarship aid  2022-23 (SFA2223)"
        ),
    ]
    return (col_mean,)


@app.cell
def _(data_df, pca_df, pl):
    joined_df = pca_df.join(data_df, on="Institution Name")
    columns_to_average = [
        col
        for col in joined_df.columns
        if col not in ["Institution Name", "Cluster"]
    ]

    # Build mean expressions
    agg_exprs = [pl.mean(col).alias(col) for col in columns_to_average]

    # Group by cluster and compute means
    cluster_means = joined_df.group_by("Cluster").agg(agg_exprs)

    cluster_means
    return agg_exprs, cluster_means, columns_to_average, joined_df


@app.cell
def _(pca_df, pl):
    pca_df.filter(pl.col("Cluster") == 1)
    return


@app.cell
def _(data_df, pca_df, pl):
    peer_cluster_df = data_df.filter(
        pl.col("Institution Name").is_in(
            pca_df.filter(pl.col("Cluster") == 7)["Institution Name"]
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
    peer_institutions
    return (
        TU_dict,
        filters,
        peer_cluster_df,
        peer_institutions,
        tolerance_range,
    )


@app.cell
def _(mo):
    mo.md(r"""## TU Target""")
    return


@app.cell
def _():
    aexpenses = {
        "Student Sucess": [
            57,
        ]
    }
    return (aexpenses,)


@app.cell
def _(data_df):
    data_df.columns
    return


@app.cell
def _(set_groups):
    def print_cols():
        for key, cols in set_groups().items():
            print(key)
            for col in cols:
                print(f"\t{col}")


    print_cols()
    return (print_cols,)


@app.cell
def _(set_groups):
    len(set_groups()["Sustainability & Efficiency"])
    return


@app.cell
def _(mo):
    col_search = mo.ui.text(label="Column")
    mo.hstack([col_search])
    return (col_search,)


@app.cell
def _(col_search, data_df):
    try:
        print(data_df.columns.index(col_search.value))
    except ValueError as err:
        pass
    return


@app.cell
def _(data_df):
    def set_groups():
        category_map = {
            "Student Success": [25, 26, 27, 100, 99, 48, 55],
            "Innovation & Research": [52, 45, 17],
            "Academic Resources": list(range(60, 67)) + [47],
            # "Access & Equity" :
            # "Career & Economic Outcomes":
            "Community Engagement": [53],
            "Sustainability & Efficiency": [42, 34, 49, 58, 59],
        }

        selected_columns = {}
        for key, column_list in category_map.items():
            selected_columns[key] = [data_df.columns[i] for i in column_list]

        return selected_columns
    return (set_groups,)


@app.cell
def _(PCA, data_df, pl, set_groups):
    def get_category_scores(df: pl.DataFrame):
        """
        Calculate the composite score of a category of columns
        """
        map = {}
        for key, group in set_groups().items():
            X = df.select(group).to_numpy()
            map[key] = PCA(n_components=1).fit_transform(X)[:, 0]

        return pl.DataFrame(map).with_columns(data_df.select("Institution Name"))


    def get_total_scores(df: pl.DataFrame) -> pl.DataFrame:
        """
        Multiply the composite score of each category and sum them to get an overall score
        """
        weights = {
            "Student Success": 0.23,
            "Access & Equity": 0.18,
            "Academic Resources": 0.15,
            "Career & Economic Outcomes": 0.14,
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
    return get_category_scores, get_total_scores


@app.cell
def _(get_category_scores, peer_institutions, pl, scaled_df):
    get_category_scores(scaled_df).filter(
        pl.col("Institution Name").is_in(peer_institutions)
    ).sort(by="Innovation & Research", descending=True).with_row_index(offset=1)
    return


@app.cell
def _(data_df, get_category_scores, pl, scaled_df, set_groups):
    def global_rankings():
        for k in set_groups().keys():
            height = data_df.height
            print(
                f"{k}: {get_category_scores(scaled_df).sort(by=k, descending=True).with_row_index(offset=1).filter(pl.col('Institution Name') == 'Towson University')['index'][0]} out of {height}"
            )


    global_rankings()
    return (global_rankings,)


@app.cell
def _(get_category_scores, peer_institutions, pl, scaled_df, set_groups):
    def peer_rankings():
        for z in set_groups().keys():
            height = len(peer_institutions)
            filtered_df = (
                get_category_scores(scaled_df)
                .filter((pl.col("Institution Name").is_in(peer_institutions)))
                .sort(by=z, descending=True)
                .with_row_index(offset=1)
                .filter((pl.col("Institution Name") == "Towson University"))
            )

            print(f"{z}: {filtered_df['index'][0]} out of {height}")


    peer_rankings()
    return (peer_rankings,)


@app.cell
def _(data_df):
    column_ranges = {
        "Expenditures": (46, 61),
        "Perc_Expense": (46, 53),
    }

    selected_columns = {}
    for key, (start, end) in column_ranges.items():
        selected_columns[key] = data_df.columns[start:end]
    return column_ranges, end, key, selected_columns, start


@app.cell
def _(data_df, get_category_scores, scaled_df, set_groups):
    data_df.join(get_category_scores(scaled_df), on="Institution Name").select(
        set_groups()["Innovation & Research"]
        + ["Innovation & Research", "Institution Name"]
    ).describe()
    return


@app.cell
def _(
    data_df,
    get_category_scores,
    peer_institutions,
    pl,
    scaled_df,
    set_groups,
):
    data_df.filter(pl.col("Institution Name").is_in(peer_institutions)).join(
        get_category_scores(scaled_df), on="Institution Name"
    ).select(
        set_groups()["Innovation & Research"]
        + ["Innovation & Research", "Institution Name"]
    ).sort(by="Innovation & Research", descending=True).with_row_index(offset=1)
    return


@app.cell
def _(pl, total_pop_idx):
    def add_race_perc(data_df):
        race_cols = [
            "American Indian or Alaska Native total (EF2023A  All students total)",
            "Asian total (EF2023A  All students total)",
            "Black or African American total (EF2023A  All students total)",
            "Hispanic total (EF2023A  All students total)",
            "Native Hawaiian or Other Pacific Islander total (EF2023A  All students total)",
            "White total (EF2023A  All students total)",
            "Two or more races total (EF2023A  All students total)",
        ]

        # Initialize the pop_breakdown list
        pop_breakdown = []

        # Get the column name for the total population column
        total_pop_col = data_df.columns[total_pop_idx]

        # Loop through each race column index
        for col in race_cols:
            # Get the column name

            # Create a new column name for the percentage
            new_col_name = f"{col}_pct"

            # Create the polars expression for division
            expression = (
                pl.col(col) / pl.col("Grand total (EF2023  All students total)")
            ).alias(new_col_name)
            # Add the expression to the list
            pop_breakdown.append(expression)

        return data_df.with_columns(pop_breakdown)
    return (add_race_perc,)


@app.cell
def _(alt, pl):
    def bar_chart(df: pl.DataFrame):
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
                    sort=alt.EncodingSortField(field="Ranking", order="ascending"),
                ).axis(labelAngle=-32),
                y=alt.Y("value:Q", title="Value"),
                xOffset="variable:N",
                color=alt.Color("variable:N", title="Category"),
                tooltip=["Institution Name", "variable", "value"],
            )
            .properties(width=1500, height=900)
        )
    return (bar_chart,)


@app.cell
def _(PCA, alt, pl, scaled_df, set_groups):
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


    plot_pca_loadings(scaled_df, "Student Success", set_groups()["Student Success"])
    return plot_pca_loadings, plt, sns


@app.cell
def _(PCA, alt, pl):
    def cluster_loadings(df):
        X = df.to_numpy()
        pca = PCA(n_components=2)
        pca.fit(X)
    
        loadings = pca.components_[0]
        loading_df = pl.DataFrame({"Feature": df.columns, "Loading": loadings})
    
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
                title=f"PCA Loadings", width=600, height=400
            )
        )
        return chart

    return (cluster_loadings,)


@app.cell
def _(scaled_df):
    scaled_df.to_numpy()
    return


@app.cell
def _(cluster_loadings, scaled_df):
    cluster_loadings(scaled_df)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

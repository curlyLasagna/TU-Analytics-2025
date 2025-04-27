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
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    return KMeans, PCA, StandardScaler, alt, cs, mo, pl


@app.cell
def _(pl):
    def read_data():
        labels_df = (
            pl.read_csv("Labels.csv")
            .filter(~pl.col("VariableName").str.starts_with("State"))
            .with_columns(pl.col("Value").cast(pl.Int64))
        )

        data_df = (
            pl.read_csv("data.csv")
            .drop(
                # Redundant columns
                "UnitID",
                "Institution (entity) name (HD2023)",
                "Grand total (EF2023A  All students total)",
            )
            # Filter out non 4-year schools
            .filter(
                pl.col(
                    "Carnegie Classification 2021: Undergraduate Profile (HD2023)"
                )
                > 3
            )
        )

        mapping_dict = {
            var: dict(zip(sub_df["Value"], sub_df["ValueLabel"]))
            for var, sub_df in labels_df.group_by("VariableName")
        }

        # Decoded categorical columns
        return data_df.with_columns(
            [
                pl.col(col).cast(pl.Utf8).replace(mapping)
                for col, mapping in mapping_dict.items()
            ]
        )
    return (read_data,)


@app.cell
def _(pl):
    def list_majority_null_cols(df):
        # Total number of rows in your DataFrame
        row_count = df.height

        # Get null count for each column
        null_counts = df.select(
            [pl.col(col).is_null().sum().alias(col) for col in df.columns]
        )

        majority_null_cols = null_counts.unpivot(
            variable_name="column",
            value_name="null_count",
            # Drop columns that have more than than 1/1.7 columns mising
        ).filter(pl.col("null_count") > (row_count / 1.7))

        return majority_null_cols["column"].to_list()
    return (list_majority_null_cols,)


@app.cell
def _(StandardScaler, cs, pl, read_data):
    def preProcess_drop_majority_nulls(df):
        # Total number of rows in your DataFrame
        row_count = df.height

        # Get null count for each column
        null_counts = df.select(
            [pl.col(col).is_null().sum().alias(col) for col in df.columns]
        )

        majority_null_cols = null_counts.unpivot(
            variable_name="column",
            value_name="null_count",
            # Drop columns that have more than than 1/1.7 columns mising
        ).filter(pl.col("null_count") > (row_count / 1.7))

        to_drop = majority_null_cols["column"].to_list()

        return df.drop(to_drop)


    def preProcess_get_standardScale_df(df: pl.DataFrame):
        # Normalize data using z-score
        scaler = StandardScaler()
        features = df.select(cs.numeric())
        scaled_features = scaler.fit_transform(features.to_numpy())
        return pl.DataFrame(scaled_features, schema=features.columns)


    def preProcess_get_skewed_cols(df: pl.DataFrame):
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


    def preProcess_remove_highly_skewed_cols(df: pl.DataFrame):
        target_cols = (
            preProcess_get_skewed_cols(df)
            .filter(pl.col("skewness") > 10)["column"]
            .to_list()
        )
        return df.drop(target_cols)


    def preProcess_fill_null_median(df: pl.DataFrame) -> pl.DataFrame:
        # Fill in null values in columns with their respective medians
        features = df.select(cs.numeric()).columns
        return df.with_columns(
            pl.col(features).fill_null(pl.col(features).median())
        )


    def preProcess_fill_null_mean(df: pl.DataFrame):
        features = df.select(cs.numeric()).columns
        # Impute missing values with the mean of the column
        return df.with_columns(pl.col(features).fill_null(pl.col(features).mean()))


    def preProcess_log_transform(df, columns) -> pl.DataFrame:
        # Log transformation to alleviate skewed data
        df = df.with_columns(
            [
                pl.when(pl.col(col) <= 0)
                # Reduce accuracy but keep it mathematically correct
                .then(1e-10)
                .otherwise(pl.col(col))
                .log1p()
                .alias(f"{col}")
                for col in columns
            ]
        )
        return df


    def get_processed_df(fill_null_method: str) -> pl.DataFrame:
        """
        Return dataframe with all of the pre-processing applied
        """
        if fill_null_method == "mean":
            data = preProcess_fill_null_mean(
                preProcess_drop_majority_nulls(read_data())
            )
        else:
            data = preProcess_fill_null_median(
                preProcess_drop_majority_nulls(read_data())
            )
        institution_name = data.select("Institution Name")
        skewed_cols = preProcess_get_skewed_cols(data)["column"]
        data = preProcess_get_standardScale_df(
            preProcess_log_transform(data, skewed_cols)
        ).with_columns(institution_name)
        return data
    return (
        get_processed_df,
        preProcess_drop_majority_nulls,
        preProcess_fill_null_mean,
        preProcess_fill_null_median,
        preProcess_get_skewed_cols,
        preProcess_get_standardScale_df,
        preProcess_log_transform,
        preProcess_remove_highly_skewed_cols,
    )


@app.cell
def _(alt, pl):
    def plot_skew_bar(df: pl.DataFrame, feature: str) -> alt.Chart:
        # Assume 'Institution Name' column is present
        institution_col = "Institution Name"

        # Filter out zero values and convert to pandas
        df_filtered = df.filter(pl.col(feature) != 0)
        data_pd = df_filtered.select([institution_col, feature]).to_pandas()
        data_pd = data_pd.sort_values(by=feature, ascending=True)

        # Build bar chart
        chart = (
            alt.Chart(data_pd)
            .mark_bar()
            .encode(
                x=alt.X(
                    f"{institution_col}:N",
                    sort=data_pd[institution_col].tolist(),
                    axis=None,  # 👈 hides the x-axis label and ticks
                ),
                y=alt.Y(f"{feature}:Q", title=f"{feature}"),
                tooltip=[institution_col, feature],
            )
            .properties(width=800, height=400, title=f"{feature} by Institution")
        )
        return chart
    return (plot_skew_bar,)


@app.cell
def _(KMeans, PCA, alt, data_df, n_clusters, pca_res, pl, scaled_df):
    def visualize_elbow(cluster_range):
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


    def analysis_PCA(scaled_features, n_components):
        # Principal component analysis for clustering
        institution_col = scaled_features.select("Institution Name")
        scaled_features = scaled_features.drop("Institution Name")
        pca = PCA(n_components=n_components)
        pca_res = pca.fit_transform(scaled_features)
        pca_df = pl.DataFrame(
            pca_res, schema=[f"PC{i}" for i in range(1, n_components + 1)]
        )
        pca_df = pca_df.with_columns(institution_col)
        return pca_df


    def get_PCA_variance(df, pca_components):
        pca = PCA(n_components=pca_components)
        pca_res = pca.fit_transform(df)
        return pca.explained_variance_ratio_


    def visualize_cluster(scaled_features):
        pca_df = analysis_PCA(scaled_features, 2)

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

        return b_chart + hline + vline
    return analysis_PCA, get_PCA_variance, visualize_cluster, visualize_elbow


@app.cell
def _():
    def get_categories() -> dict:
        import json

        with open("categories.json", "r") as f:
            categories = json.load(f)
        return categories
    return (get_categories,)


@app.cell
def _(PCA, pl):
    def get_category_scores(df: pl.DataFrame, categories: dict) -> pl.DataFrame:
        """
        Calculate the composite score of a category of columns
        """
        map = {}
        for key, group in categories.items():
            X = df.select(group).to_numpy()

            # Perform PCA
            pca = PCA(n_components=1)
            pca.fit(X)

            # Store the first principal component (the composite score)
            map[key] = pca.transform(X)[:, 0]

        # Return a new DataFrame with the calculated scores and the "Institution Name"
        return pl.DataFrame(map).with_columns(df.select("Institution Name"))


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
    return get_category_scores, get_total_scores


@app.cell
def _(PCA, alt, pl):
    def plot_pca_loadings_by_groups(
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
                title=f"PCA Loadings for {group_name}", width=1000, height=500
            )
        )
        return chart


    def plot_pca_loadings(
        df: pl.DataFrame, feature_names: list[str], n: int
    ) -> alt.Chart:
        X = df.select(feature_names).to_numpy()
        pca = PCA(n_components=2)
        pca.fit(X)

        loadings = pca.components_[0]
        loading_df = pl.DataFrame(
            {"Feature": feature_names, "Loading": loadings}
        ).top_k(k=n, by="Loading")

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
            .properties(title=f"PCA Loadings", width=1000, height=500)
        )
        return chart


    def radar_chart(df: pl.DataFrame):
        import plotly_express as px

        unpivot_df = df.drop("Institution Name").unpivot()
        fig = px.line_polar(
            unpivot_df, r="value", theta="variable", line_close=True
        )
        fig.update_traces(fill="toself")
        fig.show()
    return plot_pca_loadings, plot_pca_loadings_by_groups, radar_chart


@app.cell
def _(get_categories, get_category_scores, get_total_scores):
    def get_rankings(df):
        return (
            get_total_scores(get_category_scores(df, get_categories()))
            .sort(by="Total Score", descending=True)
            .with_row_index(offset=1)
        )
    return (get_rankings,)


@app.cell
def _(get_categories, get_category_scores, get_processed_df, radar_chart):
    def radar_national():
        data = get_category_scores(
            get_processed_df("median"), get_categories()
        ).median()
        return radar_chart(data)
    return (radar_national,)


@app.cell
def _(cs, get_processed_df, plot_pca_loadings):
    def cluster_features(n: int):
        data = get_processed_df("mean")
        return plot_pca_loadings(data, data.select(cs.numeric()).columns, n)
    return (cluster_features,)


@app.cell
def _(pl, read_data):
    def get_peer():
        peer_institutions = [
            "Central Michigan University",
            "East Carolina University",
            "East Tennessee State University",
            "Illinois State University",
            "Mississippi State University",
            "Oakland University",
            "Rowan University",
            "The University of Montana",
            "Towson University",
            "University of Louisville",
            "University of North Carolina at Charlotte",
            "University of South Alabama",
            "University of Washington-Bothell Campus",
        ]

        return read_data().filter(
            pl.col("Institution Name").is_in(peer_institutions)
        )
    return (get_peer,)


@app.cell
def _(get_categories, get_category_scores, get_processed_df, pl, radar_chart):
    def radar_TU():
        data = get_category_scores(
            get_processed_df("median"), get_categories()
        ).filter(pl.col("Institution Name") == "Towson University")
        return radar_chart(data)
    return (radar_TU,)


@app.cell
def _(get_processed_df, get_rankings, pl):
    peer_institutions = [
        "Central Michigan University",
        "East Carolina University",
        "East Tennessee State University",
        "Illinois State University",
        "Mississippi State University",
        "Oakland University",
        "Rowan University",
        "The University of Montana",
        "Towson University",
        "University of Louisville",
        "University of North Carolina at Charlotte",
        "University of South Alabama",
        "University of Washington-Bothell Campus",
    ]

    filter_peers = pl.col("Institution Name").is_in(peer_institutions)


    def get_peers():
        return get_rankings(
            get_processed_df("mean").filter(filter_peers),
        )
    return filter_peers, get_peers, peer_institutions


@app.cell
def _(radar_TU):
    radar_TU()
    return


@app.cell
def _(get_categories, get_category_scores, get_processed_df):
    def top_3_peers_avg():
        res = []
        for x in get_categories():
            get_category_scores(
                get_processed_df("median"), get_categories()
            ).top_k(k=3, by=x).select([x, "Institution Name"]).mean().write_csv(
                f"{x}.csv"
            )
    return (top_3_peers_avg,)


@app.cell
def _(get_peers):
    get_peers()
    return


@app.cell
def _(alt, pl):
    def plot_feature_distribution(df: pl.DataFrame, feature: str) -> alt.Chart:
        # Convert to Pandas for Altair
        data_pd = df.select(feature).to_pandas()

        # Build histogram
        chart = (
            alt.Chart(data_pd)
            .mark_bar()
            .encode(
                x=alt.X(
                    f"{feature}:Q",
                    bin=alt.Bin(maxbins=50),
                    title=f"{feature} values",
                ),
                y=alt.Y("count():Q", title="Frequency"),
                tooltip=[feature],
            )
            .properties(
                width=600, height=400, title=f"Distribution of '{feature}'"
            )
        )
        return chart
    return (plot_feature_distribution,)


@app.cell
def _(pl, read_data):
    def top_3_expense(data_df):
        # Schools that appear in the top 5 across categories
        list = [
            "East Tennessee State University",
            "University of South Alabama",
            "Mississippi State University",
        ]

        expense_columns = [
            col
            for col in data_df.columns
            if "expenses" in col.lower() or "expenditures" in col.lower()
        ]

        return (
            data_df.filter(pl.col("Institution Name") == "Towson University")
            .select(expense_columns)
            .mean()
        )


    top_3_expense(read_data())
    return (top_3_expense,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

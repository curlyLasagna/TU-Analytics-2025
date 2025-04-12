from thefuzz import fuzz
import polars as pl

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





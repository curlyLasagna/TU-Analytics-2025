# TU Analytics 2025 Competition 

Data exploration

 > Won 1st place! Comes as a surprise as I lack basics in statistics methodologies as you take a closer look into our analysis.

[Presentation link](https://www.canva.com/design/DAGkSSAs26U/pdzYY8Vlp7g_MXrKi65bcA/view?utm_content=DAGkSSAs26U&utm_campaign=designshare&utm_medium=link2&utm_source=uniquelinks&utlId=h1c615ed8b7)

## Problem 

The OIR would like you to create a ranking of the given institutions and explain how you decided upon this ranking. You may consider methodologies from other rankings (e.g., U.S. News, Princeton Review, etc.). You may also have more than one ranking.

Questions:

1. Identify areas in which TU excels and areas which present growth opportunities

2. Suggest directions for TU’s investments and goals

3. Based on your ranking of institutions, create a set of aspirant peer institutions for Towson University and explain why they are included in your set (such as, due to exhibiting certain characteristics which you view as important)

## Dependencies

[Install `uv`](https://docs.astral.sh/uv/getting-started/installation/)

1. `uv venv` to create a virtual environment
2. `source .venv/bin/activate` to your desired shell. If you're on Windows, do `.venv\Scripts\activate`  
3. `uv sync` to install dependencies

## Running the notebook

I've entirely given up on Jupyter notebooks. I'm officially Marimo-pilled

To run the Marimo notebook: `marimo edit main.py`

## Use of AI

- Create a function to perform fuzzy merging with external resources
- Ask "What data clean up steps are required to achieve an accurate PCA value"
- Generate visualizations
- Performing text analysis on Towson University's strategic plan for 2020-2030 to find the categories that matters most to Towson University

## Improvements

### Data pre-processing

- Use a different transformation technique for alleviating skewed data on negative values
- Check to make sure if median imputation was the right choice

### Ranking Engine 

- More in-depth understanding of what principal component analysis actually does. We'll have to [read](https://support.minitab.com/en-us/minitab/help-and-how-to/statistical-modeling/multivariate/how-to/principal-components/interpret-the-results/key-results/) up more about it 
- The explained variance ratio for the composite scores isn't optimal.
    - Could've calculated the cumulative principal components with a threshold of 80% in variance ratio

``` text
Explained variance ratio for category 'Student Success': 0.548872360939206
Explained variance ratio for category 'Equity': 0.38843904868966256
Explained variance ratio for category 'Access': 0.3372161703745355
Explained variance ratio for category 'Academic Resources': 0.388477762051527
Explained variance ratio for category 'Innovation & Research': 0.4582434387393129
Explained variance ratio for category 'Sustainability & Efficiency': 0.4069360331693733
Explained variance ratio for category 'Community Engagement': 0.9915456755768609
```

### Peers
- Chose too high of a cluster number for our K-Means clustering. Choose 8, but 4 is more optimal according to the elbow method. We could've also applied the silhoutte method

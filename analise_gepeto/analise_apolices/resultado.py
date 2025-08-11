import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import os
    from IPython.display import display, Markdown
    return (pd,)


@app.cell
def _(pd):
    df_missing_data = pd.read_csv("data/01_missing_pct.csv")
    df_missing_data
    return


if __name__ == "__main__":
    app.run()

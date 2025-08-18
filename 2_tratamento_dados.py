import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""# Importação de pacotes""")
    return


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    return mo, pd


@app.cell
def _(mo):
    mo.md(r"""# Tratamento dos dados e geração das bases organizadas""")
    return


@app.cell
def _(pd):
    df_apolice = pd.read_parquet("apolices.parquet")
    df_apolice
    return (df_apolice,)


@app.cell
def _(df_apolice):
    print("Soma de valores de apólice maior que 1.000.000")
    print(df_apolice[df_apolice["Valor do Prêmio Comercial"] > 1000000]["Valor do Prêmio Comercial"].sum())
    print()
    print("Soma de valores de apólice menor que 1.000.000")
    print(df_apolice[df_apolice["Valor do Prêmio Comercial"] < 1000000]["Valor do Prêmio Comercial"].sum())
    return


@app.cell
def _(df_apolice):
    _df = df_apolice[~df_apolice.duplicated(subset=["Apólice", "Valor do Prêmio Comercial"])]

    print("Soma de valores de apólice maior que 1.000.000")
    print(_df[_df["Valor do Prêmio Comercial"] > 1000000]["Valor do Prêmio Comercial"].sum())
    print()
    print("Soma de valores de apólice menor que 1.000.000")
    print(_df[_df["Valor do Prêmio Comercial"] < 1000000]["Valor do Prêmio Comercial"].sum())
    return


if __name__ == "__main__":
    app.run()

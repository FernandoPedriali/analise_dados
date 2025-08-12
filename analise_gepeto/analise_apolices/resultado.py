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
    import altair as alt
    return alt, mo, pd


@app.cell
def _(mo):
    mo.md(r"""# Emissões mensais""")
    return


@app.cell
def _(pd):
    df_emissoes_mensal = pd.read_csv("data/02_emissoes_mensal.csv")
    df_emissoes_mensal
    return (df_emissoes_mensal,)


@app.cell
def _(alt, df_emissoes_mensal):
    # replace _df with your data source
    _chart = (
        alt.Chart(df_emissoes_mensal)
        .mark_bar()
        .encode(
            x=alt.X(field='Data de Emissão', type='nominal', title='Meses'),
            y=alt.Y(field='qtd', type='quantitative', title='Quantidade de emissões', aggregate='mean'),
            tooltip=[
                alt.Tooltip(field='Data de Emissão', title='Meses'),
                alt.Tooltip(field='qtd', aggregate='mean', format=',.0f', title='Quantidade de emissões')
            ]
        )
        .properties(
            title='Emisões ensais',
            config={
                'axis': {
                    'grid': True
                }
            }
        )
    )
    _chart
    return


@app.cell
def _(mo):
    mo.md(r"""# Motivos de cancelamento""")
    return


@app.cell
def _(pd):
    df_top_motivos_cancel = pd.read_csv("data/05_top_motivos_cancel.csv")
    df_top_motivos_cancel
    return (df_top_motivos_cancel,)


@app.cell
def _(alt, df_top_motivos_cancel):
    # replace _df with your data source
    _chart = (
        alt.Chart(df_top_motivos_cancel)
        .mark_bar()
        .encode(
            x=alt.X(field='Motivo do Cancelamento', type='nominal', sort='-y'),
            y=alt.Y(field='count', type='quantitative'),
            tooltip=[
                alt.Tooltip(field='Motivo do Cancelamento'),
                alt.Tooltip(field='count', format=',.0f')
            ]
        )
        .properties(
            height=290,
            width='container',
            config={
                'axis': {
                    'grid': False
                }
            }
        )
    )
    _chart
    return


@app.cell
def _(mo):
    mo.md(r"""# Cancelamentos por produto""")
    return


@app.cell
def _(pd):
    df_cancel_por_produto = pd.read_csv("data/06_cancel_por_produto.csv")
    df_cancel_por_produto
    return


@app.cell
def _(mo):
    mo.md(r"""# Cancelamentos por Ramo""")
    return


@app.cell
def _(pd):
    df_cancel_por_ramo_grupo = pd.read_csv("data/07_cancel_por_ramo_grupo.csv")
    df_cancel_por_ramo_grupo
    return


@app.cell
def _(mo):
    mo.md(r"""# Cancelamentos por estado""")
    return


@app.cell
def _(pd):
    df_cancel_por_uf = pd.read_csv("data/08_cancel_por_uf.csv")
    df_cancel_por_uf
    return


@app.cell
def _(mo):
    mo.md(r"""# Quantidade de apólices por produto""")
    return


@app.cell
def _(pd):
    df_top_produtos = pd.read_csv("data/09_top_produtos.csv")
    df_top_produtos
    return


@app.cell
def _(mo):
    mo.md(r"""# Quantidade de apólices por Grupo do Ramo""")
    return


@app.cell
def _(pd):
    df_top_ramo_grupo = pd.read_csv("data/10_top_ramo_grupo.csv")
    df_top_ramo_grupo
    return


@app.cell
def _(mo):
    mo.md(r"""# Quantidade de apólices por estado""")
    return


@app.cell
def _(pd):
    df_top_uf = pd.read_csv("data/11_top_uf.csv")
    df_top_uf
    return


@app.cell
def _(mo):
    mo.md(r"""# Prêmio por produto""")
    return


@app.cell
def _(pd):
    df_premio_por_produto_top = pd.read_csv("data/12_premio_por_produto_top.csv")
    df_premio_por_produto_top
    return


@app.cell
def _(mo):
    mo.md(r"""# Prêmio por Grupo do Ramo""")
    return


@app.cell
def _(pd):
    df_premio_por_ramo_top = pd.read_csv("data/13_premio_por_ramo_top.csv")
    df_premio_por_ramo_top
    return


@app.cell
def _(mo):
    mo.md(r"""# Prêmio por Estado""")
    return


@app.cell
def _(pd):
    df_premio_por_uf_top = pd.read_csv("data/14_premio_por_uf_top.csv")
    df_premio_por_uf_top
    return


@app.cell
def _(mo):
    mo.md(r"""# Faixa etária dos segurados""")
    return


@app.cell
def _(pd):
    df_perfil_faixa_etaria = pd.read_csv("data/17_perfil_faixa_etaria.csv")
    df_perfil_faixa_etaria
    return


@app.cell
def _(mo):
    mo.md(r"""# Sinistralidade por produto""")
    return


@app.cell
def _(pd):
    df_sinistralidade_por_produto = pd.read_csv("data/21_sinistralidade_por_produto.csv")
    df_sinistralidade_por_produto
    return


@app.cell
def _(mo):
    mo.md(r"""# Sinitralidade por estado""")
    return


@app.cell
def _(pd):
    df_sinistralidade_por_uf = pd.read_csv("data/22_sinistralidade_por_uf.csv")
    df_sinistralidade_por_uf
    return


@app.cell
def _(mo):
    mo.md(r"""# Status das apólices sinistradas """)
    return


@app.cell
def _(pd):
    df_sinistralidade_por_status = pd.read_csv("data/23_sinistralidade_por_status.csv")
    df_sinistralidade_por_status
    return


if __name__ == "__main__":
    app.run()

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
    import altair as alt
    import locale

    locale.setlocale(locale.LC_TIME, "pt_BR")
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
def _(mo):
    mo.md(r"""## Verificação de valores da base""")
    return


@app.cell
def _(mo):
    mo.md(rf"""### Valores iniciais, sem tratamento""")
    return


@app.cell
def _(df_apolice, mo):
    dt_mais_antiga = df_apolice["Data de Emissão"].min().strftime("%d/%B/%Y")
    dt_mais_recente = df_apolice["Data de Emissão"].max().strftime("%d/%B/%Y")

    mo.md(
        f"""
    /// admonition | Datas de emissão
    Mais antiga: **{dt_mais_antiga}**

    Mais recente: **{dt_mais_recente}**
    ///
    """
    )
    return dt_mais_antiga, dt_mais_recente


@app.cell
def _(df_apolice, mo):
    _maior = (f"{df_apolice[df_apolice["Valor do Prêmio Comercial"] > 1000000]["Valor do Prêmio Comercial"].sum():,.2f}"
        .replace(",", "X").replace(".", ",").replace("X", ".")) 
    _menor = (f"{df_apolice[df_apolice["Valor do Prêmio Comercial"] < 1000000]["Valor do Prêmio Comercial"].sum():,.2f}"
        .replace(",", "X").replace(".", ",").replace("X", ".")) 

    mo.md(
        f"""
    /// attention | Antes da limpeza de dados
    Soma de valores de apólice maior que 1.000.000:
    R${_maior}

    Soma de valores de apólice menor que 1.000.000:
    R${_menor}
    ///
    """
    )
    return


@app.cell
def _(df_apolice, mo):
    _df = df_apolice[~df_apolice.duplicated(subset=["Apólice", "Valor do Prêmio Comercial"])]

    _maior = (f"{_df[_df["Valor do Prêmio Comercial"] > 1000000]["Valor do Prêmio Comercial"].sum():,.2f}"
        .replace(",", "X").replace(".", ",").replace("X", ".")) 
    _menor = (f"{_df[_df["Valor do Prêmio Comercial"] < 1000000]["Valor do Prêmio Comercial"].sum():,.2f}"
        .replace(",", "X").replace(".", ",").replace("X", ".")) 

    mo.md(
        f"""
    /// attention | Após remover duplicações considerando valor de apólice e prêmio
    Soma de valores de apólice maior que 1.000.000:
    R${_maior}

    Soma de valores de apólice menor que 1.000.000:
    R${_menor}
    ///
    """
    )
    return


@app.cell
def _(df_apolice, mo):
    agrupar_por = mo.ui.dropdown(options=df_apolice.columns, label="Agrupar por",searchable=True,)
    visualizar = mo.ui.multiselect(options=df_apolice.columns, label="Contra",)
    mo.vstack([agrupar_por,visualizar])
    return


@app.cell
def _(df_apolice):
    investigacao_premio_apolices = df_apolice.groupby(['Apólice'], as_index=False)['Valor do Prêmio Comercial'].sum().sort_values(by="Valor do Prêmio Comercial", ascending=False)
    investigacao_premio_apolices = investigacao_premio_apolices.merge(
        df_apolice[["Apólice", "Nome do Produto"]],
        on="Apólice",
        how="left"
    ).drop_duplicates()
    investigacao_premio_apolices
    return (investigacao_premio_apolices,)


@app.cell
def _(mo):
    qtde_apolices_maior_premio = mo.ui.slider(start=0, stop=25, step=1, value=10, show_value=True, label="Quantidade de apólices")
    return (qtde_apolices_maior_premio,)


@app.cell
def _(
    dt_mais_antiga,
    dt_mais_recente,
    investigacao_premio_apolices,
    mo,
    qtde_apolices_maior_premio,
):
    _df = investigacao_premio_apolices.head(qtde_apolices_maior_premio.value)
    valor = (f"R${_df["Valor do Prêmio Comercial"]
        .sum():,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))



    mo.md(f"""
    {qtde_apolices_maior_premio}
    ### Apólices com maior soma de valor de prêmio recebido entre *{dt_mais_antiga}* e *{dt_mais_recente}*

    {mo.ui.table(_df)}

    ### Soma de prêmio das {qtde_apolices_maior_premio.value} apólices com maior prêmio
    \n {valor}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Início do tratamento


    """
    )
    return


@app.cell
def _(df_apolice, mo):
    _valor = f"R${df_apolice["Valor do Prêmio Comercial"].sum():,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

    mo.md(f"""
        ## Verificação dos tipos de endosso
        {mo.ui.table(df_apolice["Descrição do Endosso"].value_counts().reset_index())}

        Valor de prêmio somado de toda a base: **{_valor}**

        Queremos uma lista apenas com tipos de endosso "APÓLICE" e "EMISSAO DE APÓLICE".
        Essas serão as apólices que foram emitidas no período.
    """)
    return


@app.cell
def _(df_apolice, mo):
    # Obter número de apólices realmente emitidas dentro do período
    apolices_emitidas_no_periodo = df_apolice[df_apolice["Descrição do Endosso"].isin(["APÓLICE", "EMISSAO DE APÓLICE"])]
    df_apolice_filtrado = df_apolice[df_apolice['Apólice'].isin(apolices_emitidas_no_periodo['Apólice'])]

    _valor = f"R${df_apolice_filtrado["Valor do Prêmio Comercial"].sum():,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

    mo.md(f"""
        ## Verificação dos tipos de endosso
        Agora, após remover apólices não emitidas no período, mas aindad vigentes
        {mo.ui.table(df_apolice_filtrado["Descrição do Endosso"].value_counts().reset_index())}

    """)


    return (df_apolice_filtrado,)


@app.cell
def _(df_apolice_filtrado, mo):
    # Estes números ainda estão 
    _df_emitido = df_apolice_filtrado[df_apolice_filtrado["Descrição do Endosso"].isin(["APÓLICE", "EMISSAO DE APÓLICE", "COBRANÇA", "ENDOSSO DE COBRANÇA ADICIONAL DE PREMIO"])]
    _valor_emitido = f"R${_df_emitido["Valor do Prêmio Comercial"].sum():,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

    _df_cancelado_restituido = df_apolice_filtrado[~df_apolice_filtrado["Descrição do Endosso"].isin(["APÓLICE", "EMISSAO DE APÓLICE", "COBRANÇA"])]
    _valor_cancelado_restituido = f"R${_df_cancelado_restituido["Valor do Prêmio Comercial"].sum():,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

    _valor_cancelado_restituido_vdd = df_apolice_filtrado[df_apolice_filtrado["Descrição do Endosso"].isin(["CANCELAMENTO", "RESTITUIÇÃO", "ENDOSSO DE RESTITUIÇÃO DE PREMIO", "ENDOSSO DE CANCELAMENTO COM RESTITUIÇÃO DE PREMIO"])]
    _valor_cancelado_restituido_vdd = f"R${_valor_cancelado_restituido_vdd["Valor do Prêmio Comercial"].sum():,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


    mo.md(f"""
    Prêmio emitido: {_valor_emitido}

    Prêmio cancelado ou restituído: {_valor_cancelado_restituido}

    Valor realmente cancelado ou restituido: {_valor_cancelado_restituido_vdd}
    """)
    return


@app.cell
def _(
    df_apolice_filtrado,
    dt_mais_antiga,
    dt_mais_recente,
    mo,
    qtde_apolices_maior_premio,
):
    investigacao_premio_apolices_filtrado = df_apolice_filtrado.groupby(['Apólice'], as_index=False)['Valor do Prêmio Comercial'].sum().sort_values(by="Valor do Prêmio Comercial", ascending=False)
    investigacao_premio_apolices_filtrado = investigacao_premio_apolices_filtrado.merge(
        df_apolice_filtrado[["Apólice", "Nome do Produto"]],
        on="Apólice",
        how="left"
    ).drop_duplicates()


    _df = investigacao_premio_apolices_filtrado.head(qtde_apolices_maior_premio.value)
    _valor = (f"R${_df["Valor do Prêmio Comercial"]
        .sum():,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))



    mo.md(f"""
    {qtde_apolices_maior_premio}
    ### Apólices com maior soma de valor de prêmio recebido entre *{dt_mais_antiga}* e *{dt_mais_recente}*
    (após filtragem de apólices não emitidas no período)

    {mo.ui.table(_df)}

    ### Soma de prêmio das {qtde_apolices_maior_premio.value} apólices com maior prêmio
    \n {_valor}
    """)
    return


if __name__ == "__main__":
    app.run()

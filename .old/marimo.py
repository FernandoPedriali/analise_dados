import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():
    # Importação de pacotes
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    return (pd,)


@app.cell
def _(pd):
    # Importação dos dados de apólices
    df_apolice = pd.read_excel('Apolices_Emitidas_.xlsx')

    # Ajustar nome de colunas
    df_apolice_colunas_nomes ={
    	"nrProposta":						"Proposta",
    	"nrApolice":						"Apólice",
    	"nrEndosso":						"Endosso",
    	"nrEndossoAssociado":				"Endosso Associado",
    	"dsEndosso":						"Descrição do Endosso",
    	"statusApolice":					"Status da Apólice",
    	"idGrupoRamoSeguro":				"ID do Grupo do Ramo",
    	"RamoSeguro":						"Número do Ramo",
    	"nmGrupoRamoSeguro":				"Nome do Grupo do Ramo",
    	"dtEmissaoMovimento":				"Data de Emissão",
    	"dtIniVigencia":					"Data de Início da Vigência",
    	"dtFimVigencia":					"Data de Fim da Vigência",
    	"vlPremioComercialSeguroMoeda":		"Valor do Prêmio Comercial",
    	"vlCustoAquisicaoOperacionalMoeda":	"Valor do Custo de Aquisição",
    	"vlIOF":							"Valor do IOF",
    	"vlAdicionalFracionamentoMoeda":	"Valor do Adicional de Fracionamento",
    	"vlIS":								"Valor de IS",
    	"nrTotalFracionamento":				"Número Total de Fracionamentos",
    	"nrCpfCnpjCorretorLider":			"CPF/CNPJ do Corretor Líder",
    	"nmRazaoSocialCorretorLider":		"Razão Social do Corretor Líder",
    	"dtNnascimento":					"Data de Nascimento",
    	"nmSexo":							"Sexo",
    	"tipoPessoa":						"Tipo de Pessoa",
    	"cdProdutoApolice":					"Código do Produto",
    	"nmProduto":						"Nome do Produto",
    	"nmCep":							"CEP",
    	"nmCidade":							"Cidade",
    	"nmUF":								"UF"
    }
    df_apolice = df_apolice.rename(columns=df_apolice_colunas_nomes)
    memory_usage_before_df_apolice = df_apolice.memory_usage(deep=True)
    types_before = df_apolice.dtypes

    # Ajustar tipos de colunas
    df_apolice_colunas_tipos ={
    	"Descrição do Endosso":					"category",
    	"Status da Apólice":					"category",
    	"ID do Grupo do Ramo":					"category",
    	"Número do Ramo":						"category",
    	"Nome do Grupo do Ramo":				"category",
    	"Data de Emissão":						"datetime64[ns]",
    	"Data de Início da Vigência":			"datetime64[ns]",
    	"Data de Fim da Vigência":				"datetime64[ns]",
    	"Valor do Prêmio Comercial":			"float64",
    	"Valor do Custo de Aquisição":			"float64",
    	"Valor do IOF":							"float64",
    	"Valor do Adicional de Fracionamento":	"float64",
    	"Valor de IS":							"float64",
    	"Número Total de Fracionamentos":		"category",
    	"CPF/CNPJ do Corretor Líder":			"category",
    	"Razão Social do Corretor Líder":		"category",
    	"Data de Nascimento":					"datetime64[ns]",
    	"Sexo":									"category",
    	"Tipo de Pessoa":						"category",
    	"Código do Produto":					"category",
    	"Nome do Produto":						"category",
    	"CEP":									"string",
    	"Cidade":								"category",
    	"UF":									"category"
    	}
    df_apolice = df_apolice.astype(df_apolice_colunas_tipos)
    memory_usage_after_df_apolice = df_apolice.memory_usage(deep=True)
    types_after = df_apolice.dtypes

    # Gerar parquet
    #df_apolice.to_parquet("apolices.parquet", index=False)
    return (
        df_apolice,
        memory_usage_after_df_apolice,
        memory_usage_before_df_apolice,
        types_after,
        types_before,
    )


@app.cell
def _(df_apolice):
    df_apolice
    return


@app.cell
def _(
    memory_usage_after_df_apolice,
    memory_usage_before_df_apolice,
    pd,
    types_after,
    types_before,
):
    # Comparar uso de memória de apólices
    memory_usage_df_apolice = pd.DataFrame(
        {
            "types_before": types_before,
            "types_after": types_after,
            "Memory Usage Before": memory_usage_before_df_apolice,
            "Memory Usage After": memory_usage_after_df_apolice,
            "Percentage Difference": (
                memory_usage_after_df_apolice - memory_usage_before_df_apolice
            )
            / memory_usage_before_df_apolice
            * 100,
        }
    )
    memory_usage_df_apolice.loc["Diferença total de memória"] = [
        "---",
        "---",
        memory_usage_df_apolice["Memory Usage Before"].sum(),
        memory_usage_df_apolice["Memory Usage After"].sum(),
        (
            memory_usage_df_apolice["Memory Usage After"].sum()
            - memory_usage_df_apolice["Memory Usage Before"].sum()
        )
        / memory_usage_df_apolice["Memory Usage Before"].sum()
        * 100,
    ]

    memory_usage_df_apolice
    return


if __name__ == "__main__":
    app.run()

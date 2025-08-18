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
    mo.md(r"""# Importação dos dados e geração dos arquivos .parquet""")
    return


@app.cell
def _(mo):
    mo.md(r"""## Apólices""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Sinistros""")
    return


@app.cell
def _(pd):
    # Importação dos dados de apólices
    _df_apolice = pd.read_excel('Apolices_Emitidas_new.xlsx')

    # Ajustar nome de colunas
    _df_apolice_colunas_nomes ={
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
    	"nmUF":								"UF",
        "cd_usuario_venda":					"PA Venda",
        "nm_motivo":							"Motivo de cancelamento"
    }
    _df_apolice = _df_apolice.rename(columns=_df_apolice_colunas_nomes)
    memory_usage_before_df_apolice = _df_apolice.memory_usage(deep=True)
    types_before_apolice = _df_apolice.dtypes

    # Ajustar tipos de colunas
    _df_apolice_colunas_tipos ={
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
    	"UF":									"category",
        "PA Venda":								"category",
        "Motivo de cancelamento":				"category"
    	}
    _df_apolice = _df_apolice.astype(_df_apolice_colunas_tipos)
    memory_usage_after_df_apolice = _df_apolice.memory_usage(deep=True)
    types_after_apolice = _df_apolice.dtypes

    # Gerar parquet
    _df_apolice.to_parquet("apolices.parquet", index=False)
    return (
        memory_usage_after_df_apolice,
        memory_usage_before_df_apolice,
        types_after_apolice,
        types_before_apolice,
    )


@app.cell
def _(
    memory_usage_after_df_apolice,
    memory_usage_before_df_apolice,
    pd,
    types_after_apolice,
    types_before_apolice,
):
    # Comparar uso de memória de apólices
    memory_usage_df_apolice = pd.DataFrame({
        'types_before': types_before_apolice,
    	'types_after': types_after_apolice,
        'Memory Usage Before': memory_usage_before_df_apolice,
        'Memory Usage After': memory_usage_after_df_apolice,
        'Percentage Difference': (memory_usage_after_df_apolice
                                  - memory_usage_before_df_apolice)
                                  / memory_usage_before_df_apolice
    })
    memory_usage_df_apolice.loc['Diferença total de memória'] = [
        '---',
        '---',
        memory_usage_df_apolice['Memory Usage Before'].sum(),
        memory_usage_df_apolice['Memory Usage After'].sum(),
        (memory_usage_df_apolice['Memory Usage After'].sum() 
         - memory_usage_df_apolice['Memory Usage Before'].sum()) 
         / memory_usage_df_apolice['Memory Usage Before'].sum() * 100
    ]

    memory_usage_df_apolice
    return


@app.cell
def _(pd):
    # Importação dos dados de sinistros
    _df_sinistro = pd.read_excel('Apolices_Sinistradas_Sicoob_.xlsx')

    # Ajustar nome de colunas
    _df_sinistro_colunas_nomes ={
    	"nrApolice":							"Apólice",
    	"nrEndosso":							"Endosso",
    	"dsCoberturaSeguro":					"Nome da Cobertura",
    	"CdCoberturaSeguro":					"Código da Cobertura",
    	"idGrupoRamoSeguro":					"ID do Grupo do Ramo",
    	"idRamoSeguro":							"ID do Ramo",
    	"nrSinistro":							"Número do Sinistro",
    	"MovimentoSinistro":					"Nome do Movimento de Sinistro",
    	"dtEmissaoEventoSinistro":				"Data de Emissão do Movimento de Sinistro",
    	"vlEventoSinistroMoeda":				"Valor do Evento de Sinistro (Moeda)",
    	"vlEventoSinistroBRL":					"Valor do Evento de Sinistro",
    	"dtOcorrenciaSinistro":					"Data de Ocorrência do Sinistro",
    	"dtAvisoSinistro":						"Data de Aviso do Sinistro",
    	"dtRegistroSinistro":					"Data de Registro do Sinistro",
    	"cdUFOcorrenciaSinistro":				"UF da Ocorrência do Sinistro",
    	"nrCpfCnpjContraparte":					"CPF/CNPJ da Contraparte",
    	"nmRazaoSocialContraparte":				"Razão Social da Contraparte",
    	"cdProcessoJudicial":					"Código do Processo Judicial",
    }
    _df_sinistro = _df_sinistro.rename(columns=_df_sinistro_colunas_nomes)
    memory_usage_before_df_sinistro = _df_sinistro.memory_usage(deep=True)
    types_before_sinistro = _df_sinistro.dtypes

    # Ajustar tipos de colunas
    _df_sinistro_colunas_tipos ={
    	"Nome da Cobertura":						"category",
    	"Código da Cobertura":						"category",
    	"ID do Grupo do Ramo":						"category",
    	"ID do Ramo":								"category",
    	"Nome do Movimento de Sinistro":			"category",
    	"Data de Emissão do Movimento de Sinistro":	"datetime64[ns]",
    	"Valor do Evento de Sinistro (Moeda)":		"float64",
    	"Valor do Evento de Sinistro":				"float64",
    	"Data de Ocorrência do Sinistro":			"datetime64[ns]",
    	"Data de Aviso do Sinistro":				"datetime64[ns]",
    	"Data de Registro do Sinistro":				"datetime64[ns]",
    	"UF da Ocorrência do Sinistro":				"category",
    	"CPF/CNPJ da Contraparte":					"float64",
    }
    _df_sinistro_tipo = _df_sinistro.astype(_df_sinistro_colunas_tipos)
    memory_usage_after_df_sinistro = _df_sinistro_tipo.memory_usage(deep=True)
    types_after_sinistro = _df_sinistro.dtypes

    # Exportar para parquet
    _df_sinistro_tipo.to_parquet("sinistros.parquet", index=False)
    return (
        memory_usage_after_df_sinistro,
        memory_usage_before_df_sinistro,
        types_after_sinistro,
        types_before_sinistro,
    )


@app.cell
def _(
    memory_usage_after_df_sinistro,
    memory_usage_before_df_sinistro,
    pd,
    types_after_sinistro,
    types_before_sinistro,
):
    # Comparar uso de memória de sinistro
    memory_usage_df_sinistro = pd.DataFrame({
        'types_before': types_before_sinistro,
    	'types_after': types_after_sinistro,
        'Memory Usage Before': memory_usage_before_df_sinistro,
        'Memory Usage After': memory_usage_after_df_sinistro,
        'Percentage Difference': (memory_usage_after_df_sinistro
                                  - memory_usage_before_df_sinistro)
                                  / memory_usage_before_df_sinistro * 100
    })
    memory_usage_df_sinistro.loc['Diferença total de memória'] = [
        '---',
        '---',
        memory_usage_df_sinistro['Memory Usage Before'].sum(),
        memory_usage_df_sinistro['Memory Usage After'].sum(),
        (memory_usage_df_sinistro['Memory Usage After'].sum() 
         - memory_usage_df_sinistro['Memory Usage Before'].sum()) 
         / memory_usage_df_sinistro['Memory Usage Before'].sum() * 100
    ]

    memory_usage_df_sinistro
    return


if __name__ == "__main__":
    app.run()

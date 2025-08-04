import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------
# Função de gráfico de Pareto
# ------------------------------
def plot_pareto(
    df: pd.DataFrame,
    agrupa_por: str,
    campo_agrupado: str,
    tipo_agg: str,
    ds_tipo_agg: str
) -> tuple:
    df_acumulado = df.groupby(agrupa_por, observed=True).agg({
        campo_agrupado: tipo_agg
    }).reset_index()

    df_acumulado.sort_values(by=campo_agrupado, ascending=False, inplace=True)

    df_acumulado['per_do_todo'] = df_acumulado[campo_agrupado] / df_acumulado[campo_agrupado].sum() * 100
    df_acumulado['per_accum'] = df_acumulado['per_do_todo'].cumsum()
    df_acumulado['% do todo'] = df_acumulado['per_do_todo'].apply(lambda x: f"{x:.2f}%")
    df_acumulado['% acumulada'] = df_acumulado['per_accum'].apply(lambda x: f"{x:.2f}%")

    pareto1 = df_acumulado.loc[df_acumulado['per_accum'] > 80].iloc[0]['per_accum']
    pareto2 = len(df_acumulado.loc[df_acumulado['per_accum'] <= pareto1])
    pareto3 = len(df_acumulado)
    
    st.subheader("Resultados da Análise de Pareto")
    st.info(f"{(pareto2 / pareto3) * 100:.2f}% das categorias de '{agrupa_por}' correspondem a {pareto1:.2f}% do total de {ds_tipo_agg} '{campo_agrupado}'")
            

    # Agrupar categorias menores como "Demais" se houver mais de 10 categorias
    if len(df_acumulado) > 15:
        top = df_acumulado.iloc[:50].copy()
        restantes = df_acumulado.iloc[50:]
        total_restante = restantes[campo_agrupado].sum()

        linha_demais = pd.DataFrame({
            agrupa_por: ['Demais'],
            campo_agrupado: [total_restante]
        })

        df_acumulado_resumido = pd.concat([top, linha_demais], ignore_index=True)


    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.bar(df_acumulado_resumido[agrupa_por].astype(str), df_acumulado_resumido[campo_agrupado], color='skyblue')
    ax1.set_xlabel(agrupa_por)
    ax1.set_ylabel(f'{tipo_agg} de {campo_agrupado}', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.tick_params(axis='x', rotation=90)

    ax2 = ax1.twinx()
    ax2.plot(df_acumulado_resumido[agrupa_por].astype(str), df_acumulado_resumido['per_accum'], color='red', marker='o')
    ax2.set_ylabel('% Acumulada', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(0, 110)

    plt.title(f'Gráfico de Pareto - {campo_agrupado} por {agrupa_por}')
    ax1.grid(True, axis='y', linestyle='--', alpha=0.7)

    for i, val in enumerate(df_acumulado_resumido['per_accum']):
        ax2.annotate(f'{val:.2f}%', (df_acumulado_resumido[agrupa_por].astype(str).iloc[i], val),
                     textcoords="offset points", xytext=(0, 5), ha='center',
                     fontsize=8, color='red')

    return fig, df_acumulado.drop(columns=['per_do_todo', 'per_accum'])


# ------------------------------
# Início do app Streamlit
# ------------------------------
st.set_page_config(page_title="Análise de Pareto", layout="wide")
st.title("📊 Dados de apólices e sinistros")

# Upload da configuração
try:
    df_config = pd.read_csv("analises_pareto_disponiveis.csv", sep=";")
except Exception as e:
    st.error(f"Erro ao carregar arquivo analises_pareto_disponiveis.csv: {e}")
    st.stop()

if df_config.empty:
    st.warning("Nenhuma configuração disponível para análise de Pareto.")
    st.stop()

# Upload dos dados
try:
    df = pd.read_parquet("apolices.parquet")
except Exception as e:
    st.error(f"Erro ao carregar o arquivo 'apolices.parquet': {e}")
    st.stop()

st.subheader("Pré-visualização dos dados")
st.dataframe(df.head(10), use_container_width=True)
st.markdown("---")

abas = st.tabs(["📉 Análise de Pareto"])

with abas[0]:
    st.subheader("📊 Gráfico de Pareto baseado em configurações pré-definidas")

    campo_agrupado_escolhido, df_config_filtrado = st.columns([2, 1])
    with campo_agrupado_escolhido:
        campo_agrupado_escolhido = st.selectbox(
            "Selecione o campo a ser analisado",
            df_config['campo_agrupado'].unique()
        )
    with df_config_filtrado:
        df_config_filtrado = df_config[df_config['campo_agrupado'] == campo_agrupado_escolhido]
        agrupa_por_escolhido = st.selectbox(
            "Agrupar por",
            df_config_filtrado['agrupa_por'].unique()
        )

    config_linha = df_config_filtrado[df_config_filtrado['agrupa_por'] == agrupa_por_escolhido].iloc[0]
    tipo_agg = config_linha['tipo_agg']
    ds_tipo_agg = config_linha['ds_tipo_agg']

    if st.button("Gerar análise de Pareto"):
        try:
            fig, df_resultado = plot_pareto(
                df,
                agrupa_por=agrupa_por_escolhido,
                campo_agrupado=campo_agrupado_escolhido,
                tipo_agg=tipo_agg,
                ds_tipo_agg=ds_tipo_agg
            )
            st.subheader("📋 Tabela Resultante")
            st.dataframe(df_resultado, use_container_width=True)
            st.subheader("📈 Gráfico de Pareto")
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Erro ao gerar gráfico: {e}")

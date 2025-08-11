import marimo

__generated_with = "0.14.16"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Análise de Apólices e Sinistros — Notebook

    Pipeline completo (reprodutível) com preparação, análises, integração e relatório executivo.
    """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    from pathlib import Path
    import pandas as pd
    import numpy as np
    from IPython.display import Markdown, display

    INPUT_AP = Path('apolices.csv')
    INPUT_SI = Path('sinistros.csv')
    OUT_DIR = Path('/data')
    OUT_DIR.mkdir(exist_ok=True, parents=True)

    df = pd.read_csv(INPUT_AP, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    df.head(3)
    return Markdown, OUT_DIR, Path, df, display, np, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Integração com Sinistros e Export de Tabelas""")
    return


@app.cell
def _(OUT_DIR, Path, df, np, pd):
    def to_date(s):
        return pd.to_datetime(s, dayfirst=True, errors='coerce')

    def to_num_br(s):
        return pd.to_numeric(s.astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False), errors='coerce')
    _cols = {'apolice': 'Apólice', 'endosso': 'Endosso', 'desc_endosso': 'Descrição do Endosso', 'status': 'Status da Apólice', 'motivo_cancel': 'Motivo do Cancelamento', 'produto_nome': 'Nome do Produto', 'ramo_grupo': 'ID do Grupo do Ramo', 'uf': 'UF', 'cidade': 'Cidade', 'dt_emissao': 'Data de Emissão', 'dt_inicio': 'Data de Início da Vigência', 'dt_fim': 'Data de Fim da Vigência', 'sexo': 'Sexo', 'tipo_pessoa': 'Tipo de Pessoa'}
    for c in [c for c in df.columns if any((k in c.lower() for k in ['valor', 'vl_', 'prêmio', 'premio', 'iof', 'custo', 'fracionamento']))]:
        if c + ' (num)' not in df.columns:
            df[c + ' (num)'] = to_num_br(df[c])
    df['_dt_ini'] = to_date(df[_cols['dt_inicio']]) if _cols['dt_inicio'] else pd.NaT
    df['_dt_fim'] = to_date(df[_cols['dt_fim']]) if _cols['dt_fim'] else pd.NaT
    _col_premio = next((c for c in df.columns if c.endswith(' (num)') and ('prêmio' in c.lower() or 'premio' in c.lower())), None)
    col_custo = next((c for c in df.columns if c.endswith(' (num)') and 'custo' in c.lower()), None)
    si_path = Path('sinistros.csv')
    if si_path.exists():
        si = pd.read_csv(si_path, low_memory=False)
        si.columns = [c.strip() for c in si.columns]
        col_si_apolice = 'Apólice' if 'Apólice' in si.columns else next((c for c in si.columns if 'apolice' in c.lower()), None)
        col_si_valor = 'Valor do Evento de Sinistro' if 'Valor do Evento de Sinistro' in si.columns else next((c for c in si.columns if 'valor' in c.lower() and 'sinistro' in c.lower()), None)
        col_si_data_ocorr = 'Data de Ocorrência do Sinistro' if 'Data de Ocorrência do Sinistro' in si.columns else next((c for c in si.columns if 'ocorr' in c.lower()), None)
        col_si_num = 'Número do Sinistro' if 'Número do Sinistro' in si.columns else next((c for c in si.columns if 'sinistro' in c.lower() and 'número' in c.lower()), None)
        si['_dt_ocorr'] = to_date(si.get(col_si_data_ocorr))
        si['_valor_evento'] = to_num_br(si.get(col_si_valor))
        ap_base = pd.concat([df.groupby(_cols['apolice'])[_col_premio].sum(min_count=1).rename('premio_apolice') if _cols['apolice'] and _col_premio else pd.Series(dtype='float64'), df.groupby(_cols['apolice'])[col_custo].sum(min_count=1).rename('custo_apolice') if _cols['apolice'] and col_custo else pd.Series(dtype='float64'), df.groupby(_cols['apolice'])[_cols['produto_nome']].agg(lambda x: x.dropna().iloc[0] if len(x.dropna()) else np.nan).rename('produto') if _cols['apolice'] and _cols['produto_nome'] else pd.Series(dtype='object'), df.groupby(_cols['apolice'])[_cols['uf']].agg(lambda x: x.dropna().iloc[0] if len(x.dropna()) else np.nan).rename('uf') if _cols['apolice'] and _cols['uf'] else pd.Series(dtype='object'), df.groupby(_cols['apolice'])[_cols['status']].agg(lambda x: x.dropna().iloc[0] if len(x.dropna()) else np.nan).rename('status') if _cols['apolice'] and _cols['status'] else pd.Series(dtype='object'), df.groupby(_cols['apolice'])['_dt_ini'].min().rename('dt_inicio_min') if _cols['apolice'] else pd.Series(dtype='datetime64[ns]')], axis=1).reset_index()
        si_agg = si.groupby(col_si_apolice).agg(qtd_sinistros=(col_si_num, 'nunique'), primeiro_sinistro=('_dt_ocorr', 'min'), valor_sinistros_total=('_valor_evento', 'sum')).reset_index()
        _join_ap = ap_base.merge(si_agg, how='left', left_on=_cols['apolice'], right_on=col_si_apolice)
        _join_ap['tem_sinistro'] = _join_ap['qtd_sinistros'].fillna(0).gt(0)
        _join_ap['sinistralidade'] = _join_ap['valor_sinistros_total'] / _join_ap['premio_apolice']
        _join_ap['tempo_ate_primeiro_sinistro_dias'] = (_join_ap['primeiro_sinistro'] - _join_ap['dt_inicio_min']).dt.days
        _join_ap.to_csv(OUT_DIR / '20_join_por_apolice.csv', index=False)

        def summarize_by(group_col, min_pol=100):
            g = _join_ap.groupby(group_col, dropna=False).agg(apolices=(_join_ap.columns[0], 'nunique'), apolices_com_sinistro=('tem_sinistro', 'sum'), premio_total=('premio_apolice', 'sum'), sinistros_total=('valor_sinistros_total', 'sum')).assign(incidencia_sinistro=lambda d: d['apolices_com_sinistro'] / d['apolices'], sinistralidade=lambda d: d['sinistros_total'] / d['premio_total']).sort_values('sinistralidade', ascending=False)
            return g[g['apolices'] >= min_pol]
        by_produto = summarize_by('produto', 100) if 'produto' in _join_ap.columns else None
        by_uf = summarize_by('uf', 100) if 'uf' in _join_ap.columns else None
        by_status = summarize_by('status', 100) if 'status' in _join_ap.columns else None
        if by_produto is not None:
            by_produto.to_csv(OUT_DIR / '21_sinistralidade_por_produto.csv')
        if by_uf is not None:
            by_uf.to_csv(OUT_DIR / '22_sinistralidade_por_uf.csv')
        if by_status is not None:
            by_status.to_csv(OUT_DIR / '23_sinistralidade_por_status.csv')
        _valid = _join_ap['tem_sinistro'] & _join_ap['tempo_ate_primeiro_sinistro_dias'].notna() & (_join_ap['tempo_ate_primeiro_sinistro_dias'] >= 0)
        _tempo = _join_ap.loc[_valid, 'tempo_ate_primeiro_sinistro_dias']
        tempo_stats = None
        if len(_tempo):
            tempo_stats = {'qtd_validas': int(_valid.sum()), 'mediana': float(_tempo.median()), 'media': float(_tempo.mean()), 'q1': float(_tempo.quantile(0.25)), 'q3': float(_tempo.quantile(0.75)), 'pct_ate_30d': float((_tempo <= 30).mean()), 'pct_ate_90d': float((_tempo <= 90).mean())}
        tempo_stats
    return (tempo_stats,)


@app.cell
def _(Markdown, display, tempo_stats):
    for chave in tempo_stats.keys():
        chave_formadata = f'**{chave}**'
        display(Markdown(f"{chave_formadata}: {tempo_stats[chave]}")) if tempo_stats[chave] is not None else display(Markdown(f"{chave_formadata}: -"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Relatório Executivo""")
    return


@app.cell
def _(Markdown, Path, display, pd):
    OUT = Path('data')

    def read_csv(name):
        p = OUT / name
        return pd.read_csv(p, index_col=0) if p.exists() else None
    join_path = OUT / '20_join_por_apolice.csv'
    _join_ap = pd.read_csv(join_path, low_memory=False) if join_path.exists() else None
    lines = []
    if _join_ap is not None:
        apolices = _join_ap.iloc[:, 0].nunique()
        ap_sin = int(_join_ap['tem_sinistro'].sum())
        premio = float(_join_ap['premio_apolice'].sum(skipna=True))
        sin_total = float(_join_ap['valor_sinistros_total'].sum(skipna=True))
        sin_glob = sin_total / premio if premio else float('nan')
        _valid = _join_ap['tem_sinistro'] & _join_ap['tempo_ate_primeiro_sinistro_dias'].notna() & (_join_ap['tempo_ate_primeiro_sinistro_dias'] >= 0)
        _tempo = _join_ap.loc[_valid, 'tempo_ate_primeiro_sinistro_dias']
        lines.append(f'- **Apólices (únicas)**: {apolices:,}')
        lines.append(f'- **Apólices com sinistro**: {ap_sin:,}  ({ap_sin / apolices:.2%})')
        lines.append(f'- **Prêmio total**: R$ {premio:,.0f}'.replace(',', 'X').replace('.', ',').replace('X', '.'))
        lines.append(f'- **Sinistros total**: R$ {sin_total:,.0f}'.replace(',', 'X').replace('.', ',').replace('X', '.'))
        lines.append(f'- **Sinistralidade global**: {sin_glob:.2%}')
        if len(_tempo):
            lines.append(f'- **Tempo até 1º sinistro (mediana)**: {_tempo.median():.0f} dias (média {_tempo.mean():.0f} | Q1–Q3 {_tempo.quantile(0.25):.0f}–{_tempo.quantile(0.75):.0f} | ≤30d {(_tempo <= 30).mean():.2%} | ≤90d {(_tempo <= 90).mean():.2%})')
    cancel_uf = read_csv('08_cancel_por_uf.csv')
    if cancel_uf is not None and (not cancel_uf.empty):
        worst_uf = cancel_uf.sort_values('taxa_cancel', ascending=False).head(3)
        lines.append('\n**UFs com maior taxa de cancelamento**:')
        for idx, r in worst_uf.iterrows():
            lines.append(f'- {idx}: taxa {r['taxa_cancel']:.2%} (apólices {int(r['qtd'])}, canceladas {int(r['cancelados'])})')
    cancel_prod = read_csv('06_cancel_por_produto.csv')
    if cancel_prod is not None and (not cancel_prod.empty):
        worst_prod = cancel_prod.sort_values('taxa_cancel', ascending=False).head(3)
        lines.append('\n**Produtos com maior taxa de cancelamento**:')
        for idx, r in worst_prod.iterrows():
            lines.append(f'- {idx}: taxa {r['taxa_cancel']:.2%} (apólices {int(r['qtd'])}, canceladas {int(r['cancelados'])})')
    display(Markdown('\n'.join(lines) if lines else 'Sem dados para o relatório.'))
    return


@app.cell
def _(pd):
    # Carregar o CSV de apólices
    df_apolices = pd.read_csv("apolices.csv")

    # Resumo inicial do dataframe
    resumo_info = {
        "shape": df_apolices.shape,
        "colunas": df_apolices.columns.tolist(),
        "tipos": df_apolices.dtypes.astype(str).to_dict()
    }

    # Exibir apenas as primeiras linhas como amostra
    amostra = df_apolices.head(5)

    resumo_info, amostra
    return


@app.cell
def _(df, pd):
    _cols = df.columns.tolist()
    col_produto_nome = next((c for c in _cols if 'nome do produto' in c.lower()), None)
    col_ramo_grp = next((c for c in _cols if 'grupo do ramo' in c.lower()), None)
    col_uf = next((c for c in _cols if c.strip().upper() == 'UF'), None)
    col_status = next((c for c in _cols if 'status' in c.lower() and 'apólice' in c.lower() or 'apolice' in c.lower()), None)
    col_desc_endosso = next((c for c in _cols if 'descri' in c.lower() and 'endosso' in c.lower()), None)
    col_motivo_cancel = next((c for c in _cols if 'motivo' in c.lower() and 'cancel' in c.lower()), None)

    def str_contains_ci(series_name, pat):
        if series_name is None:
            return pd.Series([False] * len(df))
        return df[series_name].astype(str).str.lower().str.contains(pat, na=False)
    cancel_mask = str_contains_ci(col_status, 'cancel') | str_contains_ci(col_desc_endosso, 'cancel') | str_contains_ci(col_motivo_cancel, 'cancel')
    taxa_cancel_global = float(cancel_mask.mean())
    top_motivos = None
    if col_motivo_cancel:
        top_motivos = df.loc[cancel_mask, col_motivo_cancel].astype(str).str.strip().str.upper().value_counts().head(5).to_dict()
    uf_rank = None
    if col_uf:
        g = df.assign(x_cancel=cancel_mask.astype(int)).groupby(col_uf).agg(qtd=('x_cancel', 'size'), cancelados=('x_cancel', 'sum'))
        g['taxa_cancel'] = g['cancelados'] / g['qtd']
        g = g[g['qtd'] >= 200].sort_values('taxa_cancel', ascending=False)
        uf_rank = g.head(5)[['qtd', 'cancelados', 'taxa_cancel']].round(4).to_dict('index')
    num_cols = [c for c in df.columns if c.endswith(' (num)')]
    _col_premio = next((c for c in num_cols if 'prêmio' in c.lower() or 'premio' in c.lower()), None)
    top_produto_premio = None
    premio_total = None
    if _col_premio and col_produto_nome:
        premio_total = float(df[_col_premio].sum(skipna=True))
        top_produto_premio = df.groupby(col_produto_nome)[_col_premio].sum(min_count=1).sort_values(ascending=False).head(5).round(2).to_dict()
    prod_rank = None
    if col_produto_nome:
        g = df.assign(x_cancel=cancel_mask.astype(int)).groupby(col_produto_nome).agg(qtd=('x_cancel', 'size'), cancelados=('x_cancel', 'sum'))
        g['taxa_cancel'] = g['cancelados'] / g['qtd']
        g = g[g['qtd'] >= 100].sort_values('taxa_cancel', ascending=False)
        prod_rank = g.head(5)[['qtd', 'cancelados', 'taxa_cancel']].round(4).to_dict('index')

    def range_date(col_name):
        if col_name is None:
            return None
        s = pd.to_datetime(df[col_name], dayfirst=True, errors='coerce')
        return (str(s.min().date()) if pd.notna(s.min()) else None, str(s.max().date()) if pd.notna(s.max()) else None)
    faixas = {'emissao': range_date('Data de Emissão'), 'inicio_vigencia': range_date('Data de Início da Vigência'), 'fim_vigencia': range_date('Data de Fim da Vigência')}
    di = pd.to_datetime(df['Data de Início da Vigência'], dayfirst=True, errors='coerce')
    dfim = pd.to_datetime(df['Data de Fim da Vigência'], dayfirst=True, errors='coerce')
    de = pd.to_datetime(df['Data de Emissão'], dayfirst=True, errors='coerce')
    incons = {'vig_fim_antes_inicio': int((dfim < di).sum()), 'emissao_depois_inicio': int((de > di).sum())}
    insights_blob = {'taxa_cancel_global': taxa_cancel_global, 'top_motivos': top_motivos, 'uf_rank_cancel': uf_rank, 'prod_rank_cancel': prod_rank, 'premio_total': premio_total, 'top_produto_premio': top_produto_premio, 'faixas_data': faixas, 'inconsistencias_datas': incons}
    insights_blob
    return


if __name__ == "__main__":
    app.run()

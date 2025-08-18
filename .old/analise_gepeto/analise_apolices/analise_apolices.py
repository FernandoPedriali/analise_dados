import marimo

__generated_with = "0.14.16"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Análise de Apólices — Notebook.""")
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
    import matplotlib.pyplot as plt
    from IPython.display import Markdown, display

    # Caminhos de entrada/saída
    INPUT_AP = Path('apolices.csv')
    INPUT_SI = Path('sinistros.csv')
    OUT_DIR = Path('data')
    OUT_DIR.mkdir(exist_ok=True, parents=True)

    pd.options.display.float_format = lambda x: f'{x:,.2f}'
    return INPUT_AP, Markdown, OUT_DIR, Path, display, np, pd, plt


@app.cell
def _(OUT_DIR, Path, pd, plt):
    from typing import Optional, Tuple, Dict, Any

    def read_apolices(path: Path) -> pd.DataFrame:
        df = pd.read_csv(path, low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        return df

    def parse_date_cols(df: pd.DataFrame) -> pd.DataFrame:
        # Identificar colunas com "Data" no nome e criar colunas parseadas
        for c in [c for c in df.columns if 'data' in c.lower()]:
            df[c + ' (parsed)'] = pd.to_datetime(df[c], dayfirst=True, errors='coerce')
        return df

    def detect_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
        cols = df.columns.tolist()
        def pick(fn):
            try:
                return next((c for c in cols if fn(c)), None)
            except StopIteration:
                return None
        return {
            'apolice': pick(lambda c: c.lower() in ['apólice','apolice','nr_apolice','numero da apólice','número da apólice']),
            'endosso': pick(lambda c: c.lower() in ['endosso','nr_endosso','numero do endosso','número do endosso']),
            'desc_endosso': pick(lambda c: 'descri' in c.lower() and 'endosso' in c.lower()),
            'status': pick(lambda c: ('status' in c.lower()) and ('apólice' in c.lower() or 'apolice' in c.lower())),
            'motivo_cancel': pick(lambda c: ('motivo' in c.lower()) and ('cancel' in c.lower())),
            'produto_nome': pick(lambda c: 'nome do produto' in c.lower()),
            'produto_cod': pick(lambda c: 'código do produto' in c.lower() or 'codigo do produto' in c.lower() or 'cd_produto' in c.lower()),
            'ramo_grupo': pick(lambda c: 'grupo do ramo' in c.lower()),
            'uf': pick(lambda c: c.strip().upper() == 'UF'),
            'cidade': pick(lambda c: 'cidade' in c.lower()),
            'cep': pick(lambda c: 'cep' in c.lower()),
            'sexo': pick(lambda c: 'sexo' in c.lower()),
            'tipo_pessoa': pick(lambda c: 'tipo de pessoa' in c.lower() or 'tipo_pessoa' in c.lower()),
            'dt_emissao': pick(lambda c: 'emiss' in c.lower() and ' (parsed)' not in c.lower()),
            'dt_inicio': pick(lambda c: ('início' in c.lower() or 'inicio' in c.lower()) and ' (parsed)' not in c.lower()),
            'dt_fim': pick(lambda c: 'fim' in c.lower() and ' (parsed)' not in c.lower()),
            'dt_nasc': pick(lambda c: 'nascimento' in c.lower() or 'dt_nascimento' in c.lower())
        }

    def to_numeric_br(series: pd.Series) -> pd.Series:
        s = series.astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
        return pd.to_numeric(s, errors='coerce')

    def add_numeric_candidates(df: pd.DataFrame) -> pd.DataFrame:
        candidates = [c for c in df.columns if any(k in c.lower() for k in ['valor','vl_','prêmio','premio','iof','custo','fracionamento',' receita',' is','is '])]
        for c in candidates:
            df[c + ' (num)'] = to_numeric_br(df[c])
        return df

    def contains_ci(s: pd.Series, pat: str) -> pd.Series:
        return s.astype(str).str.lower().str.contains(pat, na=False)

    def monthly_count(s: pd.Series) -> pd.DataFrame:
        d = pd.to_datetime(s, errors='coerce')
        return d.dt.to_period('M').value_counts().sort_index().to_frame('qtd')

    def save_table(obj: Any, name: str) -> Path:
        path = OUT_DIR / f'{name}.csv'
        if obj is None:
            return path
        if isinstance(obj, pd.Series):
            obj.to_frame().to_csv(path)
        else:
            obj.to_csv(path, index=True)
        return path

    def plot_simple(series_or_df, title: str):
        # opcional — não interativo
        plt.figure()
        if isinstance(series_or_df, pd.Series):
            series_or_df.plot(kind='barh')
        else:
            series_or_df.plot(kind='barh')
        plt.title(title)
        plt.tight_layout()
        plt.show()
    return (
        Optional,
        add_numeric_candidates,
        contains_ci,
        detect_columns,
        monthly_count,
        parse_date_cols,
        read_apolices,
        save_table,
    )


@app.cell
def _(
    INPUT_AP,
    add_numeric_candidates,
    detect_columns,
    parse_date_cols,
    read_apolices,
):
    df = read_apolices(INPUT_AP)
    df = parse_date_cols(df)
    df = add_numeric_candidates(df)
    cols = detect_columns(df)

    summary_base = {
        'qtd_registros': int(len(df)),
        'qtd_colunas': int(len(df.columns)),
        'colunas_principais': cols
    }
    summary_base
    return cols, df


@app.cell
def _(cols, df, pd, save_table):
    dup_rate = None
    if cols['apolice'] and cols['endosso']:
        dup_rate = float(df.duplicated(subset=[cols['apolice'], cols['endosso']], keep=False).mean())
    missing_pct = df.isna().mean().sort_values(ascending=False)
    issues = {}
    if cols['dt_inicio'] and cols['dt_fim']:
        _di = pd.to_datetime(df[cols['dt_inicio']], dayfirst=True, errors='coerce')
        dfim = pd.to_datetime(df[cols['dt_fim']], dayfirst=True, errors='coerce')
        issues['vig_fim_antes_inicio'] = int((dfim < _di).sum())
    if cols['dt_emissao'] and cols['dt_inicio']:
        de = pd.to_datetime(df[cols['dt_emissao']], dayfirst=True, errors='coerce')
        _di = pd.to_datetime(df[cols['dt_inicio']], dayfirst=True, errors='coerce')
        issues['emissao_depois_inicio'] = int((de > _di).sum())
    save_table(missing_pct, '01_missing_pct')
    {'duplicidade_apolice_endosso_rate': dup_rate, 'inconsistencias_datas': issues}
    return


@app.cell
def _(cols, df, monthly_count, pd, save_table):
    temporal = {}
    if cols['dt_emissao']:
        temporal['emissoes_mensal'] = monthly_count(pd.to_datetime(df[cols['dt_emissao']], dayfirst=True, errors='coerce'))
        save_table(temporal['emissoes_mensal'], '02_emissoes_mensal')
    if cols['dt_inicio']:
        temporal['inicio_vigencia_mensal'] = monthly_count(pd.to_datetime(df[cols['dt_inicio']], dayfirst=True, errors='coerce'))
        save_table(temporal['inicio_vigencia_mensal'], '03_inicio_vigencia_mensal')
    if cols['dt_fim']:
        temporal['fim_vigencia_mensal'] = monthly_count(pd.to_datetime(df[cols['dt_fim']], dayfirst=True, errors='coerce'))
        save_table(temporal['fim_vigencia_mensal'], '04_fim_vigencia_mensal')

    temporal
    return


@app.cell
def _(Optional, cols, contains_ci, df, pd, save_table):
    cancel_mask = (contains_ci(df[cols['status']], 'cancel') if cols['status'] else False) | (contains_ci(df[cols['desc_endosso']], 'cancel') if cols['desc_endosso'] else False) | (contains_ci(df[cols['motivo_cancel']], 'cancel') if cols['motivo_cancel'] else False)
    df['__is_cancelado'] = cancel_mask
    cancelamento = {'taxa_global': float(cancel_mask.mean())}
    if cols['motivo_cancel']:
        top_motivos = df.loc[cancel_mask, cols['motivo_cancel']].astype(str).str.strip().str.upper().value_counts().head(15)
        cancelamento['top_motivos'] = top_motivos
        save_table(top_motivos, '05_top_motivos_cancel')
    tempo = {}
    if cols['dt_inicio']:
        _di = pd.to_datetime(df[cols['dt_inicio']], dayfirst=True, errors='coerce')
        if cols['dt_fim']:
            dt_cancel = pd.to_datetime(df[cols['dt_fim']], dayfirst=True, errors='coerce')
        elif cols['dt_emissao']:
            dt_cancel = pd.to_datetime(df[cols['dt_emissao']], dayfirst=True, errors='coerce')
        else:
            dt_cancel = pd.Series([pd.NaT] * len(df))
        delta = (dt_cancel - _di).dt.days
        tempo = {'dias_medio': float(delta.loc[cancel_mask].mean(skipna=True)) if cancel_mask.any() else None, 'dias_mediana': float(delta.loc[cancel_mask].median(skipna=True)) if cancel_mask.any() else None, 'q1_q3': (float(delta.loc[cancel_mask].quantile(0.25)), float(delta.loc[cancel_mask].quantile(0.75))) if cancel_mask.any() else None, 'pct_cancel_em_ate_30d': float((delta.loc[cancel_mask] <= 30).mean()) if cancel_mask.any() else None}
        cancelamento['tempo_ate_cancelamento'] = tempo

    def cancel_rate_table(group_col: Optional[str], min_volume=100) -> Optional[pd.DataFrame]:
        if not group_col or group_col not in df.columns:
            return None
        g = df.assign(x_cancel=df['__is_cancelado'].astype(int)).groupby(group_col, dropna=False).agg(qtd=('__is_cancelado', 'size'), cancelados=('x_cancel', 'sum'))
        g['taxa_cancel'] = g['cancelados'] / g['qtd']
        g = g[g['qtd'] >= min_volume].sort_values('taxa_cancel', ascending=False)
        return g
    tbl_prod = cancel_rate_table(cols['produto_nome'], min_volume=100)
    tbl_ramo = cancel_rate_table(cols['ramo_grupo'], min_volume=100)
    tbl_uf = cancel_rate_table(cols['uf'], min_volume=200)
    save_table(tbl_prod, '06_cancel_por_produto')
    save_table(tbl_ramo, '07_cancel_por_ramo_grupo')
    save_table(tbl_uf, '08_cancel_por_uf')
    cancelamento
    return cancel_mask, top_motivos


@app.cell
def _(cols, df, save_table):
    # %% [distribuicao]
    dist = {}
    def top_count(col, n=20):
        if not col:
            return None
        return df[col].astype(str).value_counts().head(n)

    dist['top_produtos'] = top_count(cols['produto_nome'])
    dist['top_ramo_grupo'] = top_count(cols['ramo_grupo'])
    dist['top_uf'] = top_count(cols['uf'])

    # salvar
    save_table(dist['top_produtos'], '09_top_produtos')
    save_table(dist['top_ramo_grupo'], '10_top_ramo_grupo')
    save_table(dist['top_uf'], '11_top_uf')

    dist
    return


@app.cell
def _(cols, df, save_table):
    # %% [financeiro]
    num_cols = [c for c in df.columns if c.endswith(' (num)')]
    col_premio = next((c for c in num_cols if ('prêmio' in c.lower() or 'premio' in c.lower())), None)
    col_custo = next((c for c in num_cols if 'custo' in c.lower()), None)

    financeiro = {}
    if col_premio:
        financeiro['premio_total'] = float(df[col_premio].sum(skipna=True))
        if cols['produto_nome']:
            top_premio_prod = df.groupby(cols['produto_nome'])[col_premio].sum(min_count=1).sort_values(ascending=False).head(20)
            financeiro['premio_por_produto_top'] = top_premio_prod
            save_table(top_premio_prod, '12_premio_por_produto_top')
        if cols['ramo_grupo']:
            top_premio_ramo = df.groupby(cols['ramo_grupo'])[col_premio].sum(min_count=1).sort_values(ascending=False).head(20)
            financeiro['premio_por_ramo_top'] = top_premio_ramo
            save_table(top_premio_ramo, '13_premio_por_ramo_top')
        if cols['uf']:
            top_premio_uf = df.groupby(cols['uf'])[col_premio].sum(min_count=1).sort_values(ascending=False).head(20)
            financeiro['premio_por_uf_top'] = top_premio_uf
            save_table(top_premio_uf, '14_premio_por_uf_top')

    if col_premio and col_custo:
        financeiro['margem_bruta_aprox'] = float((df[col_premio] - df[col_custo]).sum(skipna=True))

    financeiro
    return


@app.cell
def _(cols, df, np, pd, save_table):
    # %% [perfil]
    perfil = {}
    if cols['sexo']:
        perfil['sexo'] = df[cols['sexo']].astype(str).str.upper().value_counts(dropna=False)
        save_table(perfil['sexo'], '15_perfil_sexo')

    if cols['tipo_pessoa']:
        perfil['tipo_pessoa'] = df[cols['tipo_pessoa']].astype(str).str.upper().value_counts(dropna=False)
        save_table(perfil['tipo_pessoa'], '16_perfil_tipo_pessoa')

    if cols['dt_nasc']:
        dt_nasc = pd.to_datetime(df[cols['dt_nasc']], dayfirst=True, errors='coerce')
        idade = np.floor((pd.Timestamp.today() - dt_nasc).dt.days / 365.25)
        bins = [0, 18, 25, 35, 45, 55, 65, 75, 200]
        labels = ['<=18','19-25','26-35','36-45','46-55','56-65','66-75','>75']
        faixa = pd.cut(idade, bins=bins, labels=labels, right=True, include_lowest=True)
        perfil['faixa_etaria'] = faixa.value_counts().sort_index()
        save_table(perfil['faixa_etaria'], '17_perfil_faixa_etaria')

    perfil
    return


@app.cell
def _(OUT_DIR, pd, save_table):
    # %% [riscos]
    def top_cancel(table: pd.DataFrame, n=10) -> pd.DataFrame:
        if table is None or table.empty:
            return table
        return table.sort_values('taxa_cancel', ascending=False).head(n)

    riscos = {
        'produtos_maior_cancel': top_cancel(pd.read_csv(OUT_DIR/'06_cancel_por_produto.csv', index_col=0), 10) if (OUT_DIR/'06_cancel_por_produto.csv').exists() else None,
        'uf_maior_cancel': top_cancel(pd.read_csv(OUT_DIR/'08_cancel_por_uf.csv', index_col=0), 10) if (OUT_DIR/'08_cancel_por_uf.csv').exists() else None
    }

    # salvar
    if riscos['produtos_maior_cancel'] is not None:
        save_table(riscos['produtos_maior_cancel'], '18_produtos_maior_cancel')
    if riscos['uf_maior_cancel'] is not None:
        save_table(riscos['uf_maior_cancel'], '19_uf_maior_cancel')

    riscos
    return


@app.cell
def _(OUT_DIR, df):
    # %% [summary]
    resumo = {
        'registros': int(len(df)),
        'colunas': int(len(df.columns)),
        'taxa_cancel_global': float(df['__is_cancelado'].mean()) if '__is_cancelado' in df.columns else None,
        'artefatos_dir': str(OUT_DIR)
    }
    resumo
    return


@app.cell
def _(INPUT_AP, pd):
    df_apolice = pd.read_csv(INPUT_AP, low_memory=False)
    df_apolice.columns = [c.strip() for c in df_apolice.columns]
    df_apolice.head(3)
    return (df_apolice,)


@app.cell
def _(OUT_DIR, Path, df_apolice, np, pd):
    def to_date(s):
        return pd.to_datetime(s, dayfirst=True, errors='coerce')

    def to_num_br(s):
        return pd.to_numeric(s.astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False), errors='coerce')
    _cols = {'apolice': 'Apólice', 'endosso': 'Endosso', 'desc_endosso': 'Descrição do Endosso', 'status': 'Status da Apólice', 'motivo_cancel': 'Motivo do Cancelamento', 'produto_nome': 'Nome do Produto', 'ramo_grupo': 'ID do Grupo do Ramo', 'uf': 'UF', 'cidade': 'Cidade', 'dt_emissao': 'Data de Emissão', 'dt_inicio': 'Data de Início da Vigência', 'dt_fim': 'Data de Fim da Vigência', 'sexo': 'Sexo', 'tipo_pessoa': 'Tipo de Pessoa'}
    for c in [c for c in df_apolice.columns if any((k in c.lower() for k in ['valor', 'vl_', 'prêmio', 'premio', 'iof', 'custo', 'fracionamento']))]:
        if c + ' (num)' not in df_apolice.columns:
            df_apolice[c + ' (num)'] = to_num_br(df_apolice[c])
    df_apolice['_dt_ini'] = to_date(df_apolice[_cols['dt_inicio']]) if _cols['dt_inicio'] else pd.NaT
    df_apolice['_dt_fim'] = to_date(df_apolice[_cols['dt_fim']]) if _cols['dt_fim'] else pd.NaT
    _col_premio = next((c for c in df_apolice.columns if c.endswith(' (num)') and ('prêmio' in c.lower() or 'premio' in c.lower())), None)
    col_custo_2 = next((c for c in df_apolice.columns if c.endswith(' (num)') and 'custo' in c.lower()), None)
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
        ap_base = pd.concat([df_apolice.groupby(_cols['apolice'])[_col_premio].sum(min_count=1).rename('premio_apolice') if _cols['apolice'] and _col_premio else pd.Series(dtype='float64'), df_apolice.groupby(_cols['apolice'])[col_custo_2].sum(min_count=1).rename('custo_apolice') if _cols['apolice'] and col_custo_2 else pd.Series(dtype='float64'), df_apolice.groupby(_cols['apolice'])[_cols['produto_nome']].agg(lambda x: x.dropna().iloc[0] if len(x.dropna()) else np.nan).rename('produto') if _cols['apolice'] and _cols['produto_nome'] else pd.Series(dtype='object'), df_apolice.groupby(_cols['apolice'])[_cols['uf']].agg(lambda x: x.dropna().iloc[0] if len(x.dropna()) else np.nan).rename('uf') if _cols['apolice'] and _cols['uf'] else pd.Series(dtype='object'), df_apolice.groupby(_cols['apolice'])[_cols['status']].agg(lambda x: x.dropna().iloc[0] if len(x.dropna()) else np.nan).rename('status') if _cols['apolice'] and _cols['status'] else pd.Series(dtype='object'), df_apolice.groupby(_cols['apolice'])['_dt_ini'].min().rename('dt_inicio_min') if _cols['apolice'] else pd.Series(dtype='datetime64[ns]')], axis=1).reset_index()
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
    return


@app.cell
def _(Markdown, OUT_DIR, display, pd):
    def read_csv(name):
        p = OUT_DIR / name
        return pd.read_csv(p, index_col=0) if p.exists() else None
    join_path = OUT_DIR / '20_join_por_apolice.csv'
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
def _(cancel_mask, df, pd, top_motivos):
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
    cancel_mask_2 = str_contains_ci(col_status, 'cancel') | str_contains_ci(col_desc_endosso, 'cancel') | str_contains_ci(col_motivo_cancel, 'cancel')
    taxa_cancel_global = float(cancel_mask.mean())
    top_motivos_2 = None
    if col_motivo_cancel:
        top_motivos_2 = df.loc[cancel_mask, col_motivo_cancel].astype(str).str.strip().str.upper().value_counts().head(5).to_dict()
    uf_rank = None
    if col_uf:
        g = df.assign(x_cancel=cancel_mask.astype(int)).groupby(col_uf).agg(qtd=('x_cancel', 'size'), cancelados=('x_cancel', 'sum'))
        g['taxa_cancel'] = g['cancelados'] / g['qtd']
        g = g[g['qtd'] >= 200].sort_values('taxa_cancel', ascending=False)
        uf_rank = g.head(5)[['qtd', 'cancelados', 'taxa_cancel']].round(4).to_dict('index')
    num_cols_2 = [c for c in df.columns if c.endswith(' (num)')]
    _col_premio = next((c for c in num_cols_2 if 'prêmio' in c.lower() or 'premio' in c.lower()), None)
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
    dfim_2 = pd.to_datetime(df['Data de Fim da Vigência'], dayfirst=True, errors='coerce')
    de_2 = pd.to_datetime(df['Data de Emissão'], dayfirst=True, errors='coerce')
    incons = {'vig_fim_antes_inicio': int((dfim_2 < di).sum()), 'emissao_depois_inicio': int((de_2 > di).sum())}
    insights_blob = {'taxa_cancel_global': taxa_cancel_global, 'top_motivos': top_motivos, 'uf_rank_cancel': uf_rank, 'prod_rank_cancel': prod_rank, 'premio_total': premio_total, 'top_produto_premio': top_produto_premio, 'faixas_data': faixas, 'inconsistencias_datas': incons}
    insights_blob
    return


if __name__ == "__main__":
    app.run()

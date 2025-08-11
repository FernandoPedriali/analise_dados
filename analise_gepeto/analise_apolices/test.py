# Cria um notebook Jupyter (.ipynb) com a análise completa solicitada
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
cells = []

# 0) Capa / contexto
cells.append(nbf.v4.new_markdown_cell("""
# Análise de Apólices — Notebook
Este notebook organiza as análises solicitadas em etapas funcionais e reproduzíveis, sem interatividade (apenas Pandas/NumPy/Matplotlib).
"""))

# 1) Parâmetros e imports
cells.append(nbf.v4.new_code_cell("""
# %% [params]
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Caminhos de entrada/saída
INPUT_CSV = Path('/mnt/data/apolices.csv')   # ajuste se necessário
OUT_DIR = Path('/mnt/data/analise_apolices_nb')
OUT_DIR.mkdir(exist_ok=True, parents=True)

pd.options.display.float_format = lambda x: f'{x:,.2f}'
"""))

# 2) Funções utilitárias
cells.append(nbf.v4.new_code_cell("""
# %% [utils]
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
"""))

# 3) Carregar e preparar
cells.append(nbf.v4.new_code_cell("""
# %% [load + prepare]
df = read_apolices(INPUT_CSV)
df = parse_date_cols(df)
df = add_numeric_candidates(df)
cols = detect_columns(df)

summary_base = {
    'qtd_registros': int(len(df)),
    'qtd_colunas': int(len(df.columns)),
    'colunas_principais': cols
}
summary_base
"""))

# 4) Qualidade
cells.append(nbf.v4.new_code_cell("""
# %% [quality]
dup_rate = None
if cols['apolice'] and cols['endosso']:
    dup_rate = float(df.duplicated(subset=[cols['apolice'], cols['endosso']], keep=False).mean())

# Missing
missing_pct = df.isna().mean().sort_values(ascending=False)

# Datas
issues = {}
if cols['dt_inicio'] and cols['dt_fim']:
    di = pd.to_datetime(df[cols['dt_inicio']], dayfirst=True, errors='coerce')
    dfim = pd.to_datetime(df[cols['dt_fim']], dayfirst=True, errors='coerce')
    issues['vig_fim_antes_inicio'] = int((dfim < di).sum())

if cols['dt_emissao'] and cols['dt_inicio']:
    de = pd.to_datetime(df[cols['dt_emissao']], dayfirst=True, errors='coerce')
    di = pd.to_datetime(df[cols['dt_inicio']], dayfirst=True, errors='coerce')
    issues['emissao_depois_inicio'] = int((de > di).sum())

save_table(missing_pct, '01_missing_pct')

{
    'duplicidade_apolice_endosso_rate': dup_rate,
    'inconsistencias_datas': issues
}
"""))

# 5) Temporal
cells.append(nbf.v4.new_code_cell("""
# %% [temporal]
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
"""))

# 6) Cancelamentos
cells.append(nbf.v4.new_code_cell("""
# %% [cancelamentos]
cancel_mask = (
    (contains_ci(df[cols['status']], 'cancel') if cols['status'] else False) |
    (contains_ci(df[cols['desc_endosso']], 'cancel') if cols['desc_endosso'] else False) |
    (contains_ci(df[cols['motivo_cancel']], 'cancel') if cols['motivo_cancel'] else False)
)

df['__is_cancelado'] = cancel_mask

cancelamento = {'taxa_global': float(cancel_mask.mean())}

# Top motivos
if cols['motivo_cancel']:
    top_motivos = (
        df.loc[cancel_mask, cols['motivo_cancel']]
          .astype(str).str.strip().str.upper()
          .value_counts().head(15)
    )
    cancelamento['top_motivos'] = top_motivos
    save_table(top_motivos, '05_top_motivos_cancel')

# Tempo até cancelamento (aproximação)
tempo = {}
if cols['dt_inicio']:
    di = pd.to_datetime(df[cols['dt_inicio']], dayfirst=True, errors='coerce')
    if cols['dt_fim']:
        dt_cancel = pd.to_datetime(df[cols['dt_fim']], dayfirst=True, errors='coerce')
    elif cols['dt_emissao']:
        dt_cancel = pd.to_datetime(df[cols['dt_emissao']], dayfirst=True, errors='coerce')
    else:
        dt_cancel = pd.Series([pd.NaT]*len(df))
    delta = (dt_cancel - di).dt.days
    tempo = {
        'dias_medio': float(delta.loc[cancel_mask].mean(skipna=True)) if cancel_mask.any() else None,
        'dias_mediana': float(delta.loc[cancel_mask].median(skipna=True)) if cancel_mask.any() else None,
        'q1_q3': (
            float(delta.loc[cancel_mask].quantile(0.25)),
            float(delta.loc[cancel_mask].quantile(0.75))
        ) if cancel_mask.any() else None,
        'pct_cancel_em_ate_30d': float((delta.loc[cancel_mask] <= 30).mean()) if cancel_mask.any() else None
    }
    cancelamento['tempo_ate_cancelamento'] = tempo

def cancel_rate_table(group_col: Optional[str], min_volume=100) -> Optional[pd.DataFrame]:
    if not group_col or group_col not in df.columns:
        return None
    g = (
        df.assign(x_cancel=df['__is_cancelado'].astype(int))
          .groupby(group_col, dropna=False)
          .agg(qtd=('__is_cancelado','size'), cancelados=('x_cancel','sum'))
    )
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
"""))

# 7) Distribuição por produto/ramo/UF
cells.append(nbf.v4.new_code_cell("""
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
"""))

# 8) Financeiro
cells.append(nbf.v4.new_code_cell("""
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
"""))

# 9) Perfil do segurado
cells.append(nbf.v4.new_code_cell("""
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
"""))

# 10) Indicadores de risco
cells.append(nbf.v4.new_code_cell("""
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
"""))

# 11) Resumo final
cells.append(nbf.v4.new_code_cell("""
# %% [summary]
resumo = {
    'registros': int(len(df)),
    'colunas': int(len(df.columns)),
    'taxa_cancel_global': float(df['__is_cancelado'].mean()) if '__is_cancelado' in df.columns else None,
    'artefatos_dir': str(OUT_DIR)
}
resumo
"""))

nb['cells'] = cells

out_path = Path('analise_apolices/analise_apolices_notebook_test.ipynb')
out_path.parent.mkdir(exist_ok=True, parents=True)
with open(out_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

out_path.as_posix()

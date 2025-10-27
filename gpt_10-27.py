#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, re, sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    from scipy import stats
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

class Config:
    SURVEY_FILE = 'PP+survey+draft_October+16,+2025_13.32.csv'
    ELECTION_FILE = 'elections_with_reconciled_contested.csv'
    OUTPUT_DIR = Path('outputs/corrected')
    VIZ_DIR = OUTPUT_DIR / 'visualizations'
    TABLE_DIR = OUTPUT_DIR / 'tables'
    MIN_RATINGS_THRESHOLD = 10
    RATING_MAP = {
        'Very Traditional': 1,
        'Traditional': 2,
        'Not Familiar': np.nan,
        'Progressive': 3,
        'Very Progressive': 4,
    }

STATE_ABBREV_TO_NAME = {
    'AL':'Alabama','AK':'Alaska','AZ':'Arizona','AR':'Arkansas','CA':'California','CO':'Colorado',
    'CT':'Connecticut','DE':'Delaware','FL':'Florida','GA':'Georgia','HI':'Hawaii','ID':'Idaho',
    'IL':'Illinois','IN':'Indiana','IA':'Iowa','KS':'Kansas','KY':'Kentucky','LA':'Louisiana',
    'ME':'Maine','MD':'Maryland','MA':'Massachusetts','MI':'Michigan','MN':'Minnesota','MS':'Mississippi',
    'MO':'Missouri','MT':'Montana','NE':'Nebraska','NV':'Nevada','NH':'New Hampshire','NJ':'New Jersey',
    'NM':'New Mexico','NY':'New York','NC':'North Carolina','ND':'North Dakota','OH':'Ohio','OK':'Oklahoma',
    'OR':'Oregon','PA':'Pennsylvania','RI':'Rhode Island','SC':'South Carolina','SD':'South Dakota',
    'TN':'Tennessee','TX':'Texas','UT':'Utah','VT':'Vermont','VA':'Virginia','WA':'Washington',
    'WV':'West Virginia','WI':'Wisconsin','WY':'Wyoming'
}
NICKNAME = {'kim':'kimberly','bob':'robert','mike':'michael','dan':'daniel','joe':'joseph','bill':'william',
            'tom':'thomas','jim':'james','dave':'david','steve':'steven'}
def norm_first(s:str)->str:
    s = (s or '').strip().lower().replace('’',"'")
    return NICKNAME.get(s, s)
def first_ok(a:str,b:str)->bool:
    a = norm_first(a); b = norm_first(b)
    if not a or not b: return True
    return (a==b) or a.startswith(b) or b.startswith(a) or (a[0]==b[0])
def last_from_name(s:str)->str:
    s = (s or '').strip().replace('’',"'")
    return re.sub(r"[^A-Za-z'\\-]", '', s.split()[-1].lower()) if s else ''
def first_from_name(s:str)->str:
    s = (s or '').strip().replace('’',"'")
    parts = s.split()
    return re.sub(r"[^A-Za-z'\\-]", '', ' '.join(parts[:-1]).lower()) if parts else ''
def clean_jurisdiction_for_matching(j: str)->str:
    if not isinstance(j,str): return ''
    if ',' in j: j = j.rsplit(',',1)[0]
    j = re.sub(r'\\s*\\([^)]*\\)', '', j)
    j = re.sub(r'\\b(County|Parish|Borough)\\b', '', j, flags=re.I)
    j = re.sub(r'\\s+', ' ', j).strip()
    return j
def extract_state_from_jurisdiction(juris: str)->Optional[str]:
    if not isinstance(juris,str): return None
    parts = [p.strip() for p in juris.split(',')]
    if len(parts)>=2:
        abbr = parts[-1].upper()
        return STATE_ABBREV_TO_NAME.get(abbr, abbr)
    return None

class DataLoader:
    def __init__(self, survey_file: str, election_file: str):
        self.survey_file = Path(survey_file)
        self.election_file = Path(election_file)
        self.df = None
        self.questions_df = None
        self.elec = None
        self.completed_national_section = None
        self.notable_cols = []
        self.state_da_cols = []

    def load(self):
        try:
            self.df = pd.read_csv(self.survey_file, skiprows=[1])
            self.questions_df = pd.read_csv(self.survey_file, nrows=1)
            print(f"✓ Loaded survey ({len(self.df)}) with labels row")
        except Exception:
            self.df = pd.read_csv(self.survey_file)
            self.questions_df = None
            print(f"✓ Loaded survey ({len(self.df)}) (no labels row)")

        if 'con' in self.df.columns:
            self.df = self.df[self.df['con']=='Agree'].copy()

        if 'more' in self.df.columns:
            self.completed_national_section = int(self.df['more'].notna().sum())
        else:
            self.completed_national_section = len(self.df)

        try:
            self.elec = pd.read_csv(self.election_file)
            print(f"✓ Loaded elections ({len(self.elec)})")
        except Exception as e:
            print("✗ Elections load failed:", e)
            self.elec = pd.DataFrame()

        self.notable_cols = [c for c in self.df.columns if c.startswith('notable_')]

        self.state_da_cols = []
        for col in self.df.columns:
            if '_' in col and not col.startswith('Q') and not col.startswith('notable'):
                parts = col.split('_')
                if len(parts)==2 and parts[0] not in ['gender','race','Q55']:
                    if self.df[col].dropna().astype(str).str.contains('Progressive|Traditional|Not Familiar').any():
                        self.state_da_cols.append(col)

        return self

    def _parse_name_from_question(self, q_id: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        if self.questions_df is None or q_id not in self.questions_df.columns:
            return (None,None,None)
        qtext = str(self.questions_df[q_id].iloc[0])
        tail = qtext.rsplit(' - ',1)[-1].strip() if ' - ' in qtext else qtext.strip()
        if ',' in tail:
            name_part, juris_part = tail.split(',',1)
            name_part = name_part.strip(); juris = juris_part.strip().strip(',')
        else:
            name_part = tail; juris = None
        return (first_from_name(name_part) or None,
                last_from_name(name_part) or None,
                juris or None)

    def build_notable_table(self)->pd.DataFrame:
        RMAP = Config.RATING_MAP
        rows = []
        for col in self.notable_cols:
            s = self.df[col]
            numeric = s.map(RMAP).dropna()
            if numeric.empty: continue
            fname,lname,juris = self._parse_name_from_question(col)
            disp = ' '.join([x for x in [fname,lname] if x]).title() if (fname or lname) else col
            rows.append({
                'column': col,
                'name': disp,
                'fname': fname, 'lname': lname,
                'jurisdiction': juris,
                'jurisdiction_clean': clean_jurisdiction_for_matching(juris or ''),
                'state_name': extract_state_from_jurisdiction(juris or ''),
                'substantive_ratings': len(numeric),
                'total_responses': self.completed_national_section,
                'familiarity_rate': (len(numeric)/max(1,self.completed_national_section))*100.0,
                'mean_score': numeric.mean(),
                'median_score': numeric.median(),
                'std_score': numeric.std(),
                'is_notable': True
            })
        return pd.DataFrame(rows)

    def build_state_table(self)->pd.DataFrame:
        RMAP = Config.RATING_MAP
        state_to_cols: Dict[str,List[str]] = {}
        for col in self.state_da_cols:
            st = col.split('_',1)[0]
            state_to_cols.setdefault(st, []).append(col)
        engaged = {}
        for st, cols in state_to_cols.items():
            engaged[st] = int(self.df[cols].notna().any(axis=1).sum())
        rows = []
        for col in self.state_da_cols:
            st = col.split('_',1)[0]
            s = self.df[col]
            numeric = s.map(RMAP).dropna()
            if numeric.empty: continue
            fname,lname,juris = self._parse_name_from_question(col)
            disp = ' '.join([x for x in [fname,lname] if x]).title() if (fname or lname) else col
            rows.append({
                'column': col,
                'name': disp,
                'fname': fname, 'lname': lname,
                'jurisdiction': juris,
                'jurisdiction_clean': clean_jurisdiction_for_matching(juris or ''),
                'state_name': st,
                'substantive_ratings': len(numeric),
                'total_responses': engaged.get(st,0),
                'familiarity_rate': (len(numeric)/max(1,engaged.get(st,0)))*100.0,
                'mean_score': numeric.mean(),
                'median_score': numeric.median(),
                'std_score': numeric.std(),
                'is_notable': False
            })
        return pd.DataFrame(rows)

# --- Election aggregation by county token ---
def norm_txt(s: str) -> str:
    s = str(s) if s is not None else ''
    s = s.strip().lower().replace('’',"'")
    s = re.sub(r'\\s+', ' ', s)
    return s
def clean_county_token(s: str) -> str:
    s = norm_txt(s)
    s = re.sub(r'\\b(city and county of|city of|county of)\\b', '', s)
    s = s.replace(' county','').replace(' parish','').replace(' borough','')
    s = re.sub(r'\\s+', ' ', s).strip()
    return s
def counties_list_from_str(s: str):
    if pd.isna(s): return []
    parts = re.split(r'[;,|/]+', str(s))
    toks = []
    for p in parts:
        p = norm_txt(p)
        p = p.replace(' county','').replace(' parish','').replace(' borough','')
        p = re.sub(r'\\s+', ' ', p).strip()
        if p: toks.append(p)
    return toks

class ElectionAggregator:
    def __init__(self, elections: pd.DataFrame):
        self.E = elections.copy()
        self.E['state_clean'] = self.E['state'].map(norm_txt) if 'state' in self.E.columns else ''
        self.E['district_clean'] = self.E['district'].map(norm_txt) if 'district' in self.E.columns else ''
        self.E['counties_tokens'] = self.E['counties_total'].apply(counties_list_from_str) if 'counties_total' in self.E.columns else [[]]
        self.E['cand_fname_n'] = self.E['cand_fname'].map(norm_txt) if 'cand_fname' in self.E.columns else ''
        self.E['cand_lname_n'] = self.E['cand_lname'].map(norm_txt) if 'cand_lname' in self.E.columns else ''

        self.group_keys = [k for k in ['state','district','year','office'] if k in self.E.columns]
        if not self.group_keys:
            self.group_keys = [k for k in ['state','district'] if k in self.E.columns]

        self.lookup_gen = self._build_margin_lookup('vote_percent_general','winner_general')
        self.lookup_pri = self._build_margin_lookup('vote_percent_primary','winner_primary')

    
    def _build_margin_lookup(self, pct_col: str, win_col: str):
        if pct_col not in self.E.columns or win_col not in self.E.columns:
            return {}
        df = self.E[self.group_keys + ['cand_fname_n','cand_lname_n', pct_col, win_col]].copy()
        # Compute second-best percentage per race robustly
        def second_best(s):
            v = pd.to_numeric(s, errors='coerce').dropna().sort_values(ascending=False)
            return v.iloc[1] if len(v)>=2 else np.nan
        sec = df.groupby(self.group_keys)[pct_col].apply(second_best).reset_index(name='second')
        df2 = df.merge(sec, on=self.group_keys, how='left')
        cand_col = pct_col if pct_col in df2.columns else f"{pct_col}_x"
        df2['cand_pct'] = pd.to_numeric(df2.get(cand_col), errors='coerce')
        df2['second'] = pd.to_numeric(df2['second'], errors='coerce')
        df2['winner_only_margin'] = np.where(pd.to_numeric(df2[win_col], errors='coerce')==1,
                                             df2['cand_pct'] - df2['second'], np.nan)
        cand_keys = [*self.group_keys,'cand_lname_n','cand_fname_n']
        g = df2.groupby(cand_keys)['winner_only_margin'].apply(
            lambda s: np.nanmin(np.abs(pd.to_numeric(s, errors='coerce')))
        ).reset_index()
        lookup = {}
        for _, row in g.iterrows():
            key = tuple(row[k] for k in cand_keys)
            lookup[key] = row['winner_only_margin']
        return lookup


    def attach(self, df_da: pd.DataFrame) -> pd.DataFrame:
        Z = df_da.copy()
        Z['state_n'] = Z['state_name'].map(norm_txt) if 'state_name' in Z.columns else ''
        Z['jc'] = Z['jurisdiction_clean'].map(norm_txt) if 'jurisdiction_clean' in Z.columns else ''
        Z['county_token'] = Z['jc'].map(clean_county_token)
        Z['lname_n'] = Z['lname'].map(norm_txt) if 'lname' in Z.columns else ''
        Z['fname_n'] = Z['fname'].map(norm_txt) if 'fname' in Z.columns else ''
        rows = []
        for _, r in Z.iterrows():
            sub = self.E[self.E['state_clean']==r['state_n']].copy()
            if r['county_token']:
                sub = sub[[r['county_token'] in toks for toks in sub['counties_tokens']]]
            if sub.empty and r['jc']:
                sub = self.E[(self.E['state_clean']==r['state_n']) & (self.E['district_clean']==r['jc'])].copy()
            sub = sub[sub['cand_lname_n']==r['lname_n']]
            if not sub.empty and r['fname_n']:
                sub = sub[[first_ok(r['fname_n'], x) for x in sub['cand_fname_n']]]

            if sub.empty:
                rows.append({'column': r['column'], 'ever_contested_primary': np.nan, 'ever_contested_general': np.nan,
                             'winner_primary_any': np.nan, 'winner_general_any': np.nan, 'num_elections': 0,
                             'closest_primary_margin': np.nan, 'closest_general_margin': np.nan})
                continue
            pc = 'primary_contested_reconciled' if 'primary_contested_reconciled' in sub.columns else None
            gc = 'general_contested_reconciled' if 'general_contested_reconciled' in sub.columns else None
            wp = 'winner_primary' if 'winner_primary' in sub.columns else None
            wg = 'winner_general' if 'winner_general' in sub.columns else None

            ever_cp = bool(pd.to_numeric(sub[pc], errors='coerce').fillna(0).max()>0) if pc else np.nan
            ever_cg = bool(pd.to_numeric(sub[gc], errors='coerce').fillna(0).max()>0) if gc else np.nan
            win_p  = bool(pd.to_numeric(sub[wp], errors='coerce').fillna(0).max()>0) if wp else np.nan
            win_g  = bool(pd.to_numeric(sub[wg], errors='coerce').fillna(0).max()>0) if wg else np.nan

            keys = set()
            for _, rr in sub.iterrows():
                key = tuple(rr.get(k, np.nan) for k in self.group_keys) + (rr['cand_lname_n'], rr['cand_fname_n'])
                keys.add(key)

            gen_margins = [self.lookup_gen.get(k, np.nan) for k in keys]
            pri_margins = [self.lookup_pri.get(k, np.nan) for k in keys]
            gen_margins = [m for m in gen_margins if isinstance(m,(int,float)) and not np.isnan(m)]
            pri_margins = [m for m in pri_margins if isinstance(m,(int,float)) and not np.isnan(m)]
            closest_gen = float(np.nanmin(np.abs(gen_margins))) if gen_margins else np.nan
            closest_pri = float(np.nanmin(np.abs(pri_margins))) if pri_margins else np.nan

            rows.append({'column': r['column'], 'ever_contested_primary': ever_cp, 'ever_contested_general': ever_cg,
                         'winner_primary_any': win_p, 'winner_general_any': win_g, 'num_elections': int(len(sub)),
                         'closest_primary_margin': closest_pri, 'closest_general_margin': closest_gen})
        return pd.DataFrame(rows)

def contested_effect(df: pd.DataFrame, level: str)->pd.DataFrame:
    df = df.copy()
    df['ever_contested_general'] = pd.to_numeric(df['ever_contested_general'], errors='coerce')
    a = pd.to_numeric(df[df['ever_contested_general']==1]['familiarity_rate'], errors='coerce').dropna()
    b = pd.to_numeric(df[df['ever_contested_general']==0]['familiarity_rate'], errors='coerce').dropna()
    if len(a)<3 or len(b)<3 or not HAVE_SCIPY:
        return pd.DataFrame([{'level':level,'mean_contested':np.nan,'mean_uncontested':np.nan,'diff':np.nan,'t':np.nan,'p':np.nan,'n_contested':len(a),'n_uncontested':len(b)}])
    t,p = stats.ttest_ind(a,b,equal_var=False)
    sa2 = np.var(a, ddof=1); sb2 = np.var(b, ddof=1); na=len(a); nb=len(b)
    dfw = ((sa2/na + sb2/nb)**2) / ((sa2**2)/((na**2)*(na-1)) + (sb2**2)/((nb**2)*(nb-1)))
    return pd.DataFrame([{
        'level': level,
        'mean_contested': float(np.nanmean(a)),
        'mean_uncontested': float(np.nanmean(b)),
        'diff': float(np.nanmean(a)-np.nanmean(b)),
        't': float(t), 'df': float(dfw), 'p': float(p),
        'n_contested': int(na), 'n_uncontested': int(nb)
    }])

def try_ols(df: pd.DataFrame, out_csv: Path):
    try:
        import statsmodels.formula.api as smf
    except Exception:
        out_csv.write_text("statsmodels not available\n", encoding='utf-8'); return
    mdf = df[['familiarity_rate','mean_score','ever_contested_general','num_elections']].dropna()
    if len(mdf)<10:
        out_csv.write_text("insufficient rows for OLS\n", encoding='utf-8'); return
    mdf['ever_contested_general'] = mdf['ever_contested_general'].astype(int)
    model = smf.ols('familiarity_rate ~ mean_score + ever_contested_general + num_elections', data=mdf).fit()
    coefs = pd.DataFrame({'coef': model.params, 'p': model.pvalues})
    coefs.to_csv(out_csv, index=True)

def main():
    outdir = Config.OUTPUT_DIR; viz = Config.VIZ_DIR; tables = Config.TABLE_DIR
    outdir.mkdir(parents=True, exist_ok=True); viz.mkdir(parents=True, exist_ok=True); tables.mkdir(parents=True, exist_ok=True)

    dl = DataLoader(Config.SURVEY_FILE, Config.ELECTION_FILE).load()
    notable = dl.build_notable_table()
    state = dl.build_state_table()

    # Save base summaries
    notable.to_csv(outdir / 'notable_prosecutors_summary.csv', index=False)
    state.to_csv(outdir / 'state_prosecutors_summary.csv', index=False)

    # Election aggregates (county-token)
    ea = ElectionAggregator(dl.elec)
    aggN = ea.attach(notable[['column','fname','lname','jurisdiction_clean','state_name']])
    aggS = ea.attach(state[['column','fname','lname','jurisdiction_clean','state_name']])

    merged_notable = notable.merge(aggN, on='column', how='left')
    merged_state = state.merge(aggS, on='column', how='left')
    merged_notable.to_csv(outdir / 'notable_prosecutors_elections.csv', index=False)
    merged_state.to_csv(outdir / 'state_prosecutors_elections.csv', index=False)

    # Contested effects
    ceN = contested_effect(merged_notable, 'notables')
    ceS = contested_effect(merged_state, 'states')
    pd.concat([ceN, ceS], ignore_index=True).to_csv(tables / 'contested_effects.csv', index=False)

    # OLS (notables N>=10)
    mn_fit = merged_notable[merged_notable['substantive_ratings']>=Config.MIN_RATINGS_THRESHOLD].copy()
    try_ols(mn_fit, tables / 'ols_coefficients.csv')

    print("Done.")

if __name__ == '__main__':
    main()

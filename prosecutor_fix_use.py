#!/usr/bin/env python3
"""
================================================================================
PROSECUTOR IDEOLOGY & FAMILIARITY – COMPREHENSIVE MASTER (2025-10-24)
================================================================================

This script unifies the *corrected* survey pipeline with a **robust name
matching algorithm** to link survey prosecutors to election records.

Major fixes & features:
1) Rating scale is **1–4** (Very Traditional→Very Progressive). 'Not Familiar'
   is treated as missing for ideology.
2) **National familiarity denominator** = respondents who *completed* the
   national section (answered the 'more' gate).
3) **State familiarity denominator** = number of in-state respondents who
   actually engaged with their state's items (≥1 state DA response).
4) **State prosecutor name parsing fixed** (survey items are often
   "Last\tFirst\tJurisdiction").
5) **Notable prosecutor jurisdiction cleaning** - removes state codes and
   parenthetical content for better matching with election data.
6) **Best-effort hierarchical matcher**:
   - Step A: strict 4-key (state, district, first, last)
   - Step B: relaxed 3-key (state, district, last) *with first-name compatibility*
     using nickname/prefix/initial rules
   - Step C: fallback 3-key choosing winner→incumbent→most-recent year
   Each row records `match_method` and `first_name_compatible` for audit.
7) Clean exports + a small visualization set (matplotlib-only, optional).

Inputs (place in same directory):
- SURVEY_FILE  = 'PP+survey+draft_October+16,+2025_13.32.csv'
- ELECTION_FILE= 'elections_with_reconciled_contested.csv'

Outputs:
- outputs/corrected/all_prosecutors_CORRECTED.csv
- outputs/corrected/prosecutors_filtered_ideology_CORRECTED.csv
- outputs/corrected/matched_prosecutors_FULL_CORRECTED.csv
- outputs/corrected/matched_prosecutors_FILTERED_CORRECTED.csv
- outputs/corrected/matched_prosecutors_AUDIT.csv  ← contains `match_method`
- outputs/corrected/CORRECTED_FINDINGS_SUMMARY.txt
- outputs/corrected/visualizations/*.png (optional)

Run:
    python PROSECUTOR_ANALYSIS_MASTER_COMPREHENSIVE.py

Notes:
- This file avoids hard dependencies on seaborn/statsmodels: plots/models
  are skipped if unavailable.
- If your Qualtrics export uses a different header layout, the loader attempts
  a graceful fallback.
================================================================================
"""
import os
import re
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional deps (safe fallbacks)
try:
    import seaborn as sns  # noqa: F401
    _HAVE_SEABORN = True
except Exception:
    _HAVE_SEABORN = False

try:
    import statsmodels.formula.api as smf  # noqa: F401
    _HAVE_SM = True
except Exception:
    _HAVE_SM = False

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = True

# ---------------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------------
class Config:
    SURVEY_FILE = 'PP+survey+draft_October+16,+2025_13.32.csv'
    ELECTION_FILE = 'elections_with_reconciled_contested.csv'
    OUTPUT_DIR = Path('outputs/corrected')
    VIZ_DIR = OUTPUT_DIR / 'visualizations'

    MIN_RATINGS_THRESHOLD = 10

    RATING_MAP = {
        'Very Traditional': 1,
        'Traditional': 2,
        'Not Familiar': np.nan,
        'Progressive': 3,
        'Very Progressive': 4,
    }

    # 50 Notables (q_id → (Name, Jurisdiction))
    DA_NAMES: Dict[str, Tuple[str, str]] = {
        'notable_1': ('Larry Krasner', 'Philadelphia, PA'),
        'notable_2': ('Alvin Bragg', 'Manhattan, NY'),
        'notable_3': ('Mary Moriarty', 'Hennepin County, MN'),
        'notable_4': ('Brooke Jenkins', 'San Francisco, CA'),
        'notable_5': ('Chesa Boudin', 'San Francisco, CA'),
        'notable_6': ('Pamela Price', 'Alameda County, CA'),
        'notable_7': ("Nancy O'Malley", 'Alameda County, CA'),
        'notable_8': ('George Gascon', 'Los Angeles, CA'),
        'notable_9': ('Nathan Hochman', 'Los Angeles, CA'),
        'notable_10': ('Kim Foxx', 'Cook County (Chicago), IL'),
        'notable_11': ("Eileen O'Neill Burke", 'Cook County (Chicago), IL'),
        'notable_12': ('Kim Gardner', 'St Louis, MO'),
        'notable_13': ('Monique Worrell', 'Orlando, FL'),
        'notable_14': ('Andrew Warren', 'Tampa, FL'),
        'notable_15': ('Michael Dougherty', 'Boulder, CO'),
        'notable_16': ('Sim Gill', 'Salt Lake City, UT'),
        'notable_17': ('Eric Gonzalez', 'Brooklyn, NY'),
        'notable_18': ('Eli Savit', 'Washtenaw County, MI'),
        'notable_19': ('Kym Worthy', 'Wayne County (Detroit), MI'),
        'notable_20': ('Summer Stephan', 'San Diego, CA'),
        'notable_21': ('Katherine Fernandez-Rundle', 'Miami-Dade, FL'),
        'notable_22': ('Melissa Nelson', 'Duval County (Jacksonville), FL'),
        'notable_23': ('Mark Dupree', 'Wyandotte County (Kansas City), KS'),
        'notable_24': ('Jose Garza', 'Travis County (Austin), TX'),
        'notable_25': ('John Creuzot', 'Dallas County (Dallas), TX'),
        'notable_26': ('Kim Ogg', 'Harris County (Houston), TX'),
        'notable_27': ('Sean Teare', 'Harris County (Houston), TX'),
        'notable_28': ('Joe Gonzales', 'Bexar County (San Antonio), TX'),
        'notable_29': ('Rachael Rollins', 'Suffolk County (Boston), MA'),
        'notable_30': ('Kevin Hayden', 'Suffolk County (Boston), MA'),
        'notable_31': ('Ryan Mears', 'Marion County (Indianapolis), IN'),
        'notable_32': ('Dan Satterberg', 'King County (Seattle), WA'),
        'notable_33': ('Leesa Manion', 'King County (Seattle), WA'),
        'notable_34': ('Jeff Rosen', 'Santa Clara County (San Jose), CA'),
        'notable_35': ('Rachel Mitchell', 'Maricopa County (Phoenix), AZ'),
        'notable_36': ('Laura Conover', 'Pima County (Tucson), AZ'),
        'notable_37': ('Mike Schmidt', 'Multnomah County (Portland), OR'),
        'notable_38': ('Nathan Vasquez', 'Multnomah County (Portland), OR'),
        'notable_39': ('Amy Weirich', 'Shelby County (Memphis), TN'),
        'notable_40': ('Steve Mulroy', 'Shelby County (Memphis), TN'),
        'notable_41': ('Fani Willis', 'Fulton County (Atlanta), GA'),
        'notable_42': ('Sherry Boston', 'DeKalb County (Atlanta), GA'),
        'notable_43': ('Satana Deberry', 'Durham County, NC'),
        'notable_44': ('Jason Williams', 'New Orleans, LA'),
        'notable_45': ("Michael O'Malley", 'Cuyahoga County (Cleveland), OH'),
        'notable_46': ('Diana Becton', 'Contra Costa County, CA'),
        'notable_47': ('Todd Spitzer', 'Orange County, CA'),
        'notable_48': ('Anne Marie Schubert', 'Sacramento County, CA'),
        'notable_49': ('Thom LeDoux', 'Sacramento County, CA'),
        'notable_50': ('Melinda Katz', 'Queens, NY'),
    }

    US_STATE_NAMES = [
        'Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado',
        'Connecticut', 'Delaware', 'Florida', 'Georgia', 'Hawaii', 'Idaho',
        'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana',
        'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota',
        'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada',
        'New Hampshire', 'New Jersey', 'New Mexico', 'New York',
        'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma', 'Oregon',
        'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota',
        'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington',
        'West Virginia', 'Wisconsin', 'Wyoming'
    ]


# ---------------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def std_series(s: pd.Series) -> pd.Series:
    """Standardize text: lowercase, strip, collapse whitespace."""
    return (s.astype('string')
             .str.lower()
             .str.strip()
             .str.replace(r'\s+', ' ', regex=True))


STATE_ABBREV_TO_NAME = {
    'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas',
    'CA': 'California', 'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware',
    'FL': 'Florida', 'GA': 'Georgia', 'HI': 'Hawaii', 'ID': 'Idaho',
    'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa', 'KS': 'Kansas',
    'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland',
    'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi',
    'MO': 'Missouri', 'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada',
    'NH': 'New Hampshire', 'NJ': 'New Jersey', 'NM': 'New Mexico', 'NY': 'New York',
    'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio', 'OK': 'Oklahoma',
    'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina',
    'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah',
    'VT': 'Vermont', 'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia',
    'WI': 'Wisconsin', 'WY': 'Wyoming', 'DC': 'District of Columbia'
}


def state_from_jurisdiction(juris_str: str) -> Optional[str]:
    """
    Extract state from strings like 'City, ST' or 'County (City), ST'.
    Converts state abbreviations to full names to match election data format.
    """
    if not isinstance(juris_str, str):
        return None
    parts = [p.strip() for p in juris_str.split(',')]
    if len(parts) >= 2:
        state_abbrev = parts[-1].strip().upper()
        # Convert abbreviation to full name
        return STATE_ABBREV_TO_NAME.get(state_abbrev, state_abbrev)
    return None


def clean_jurisdiction_for_matching(juris_str: str) -> str:
    """
    Clean jurisdiction string for matching with election data.
    - Removes state codes (e.g., ', PA', ', NY')
    - Removes parenthetical content (e.g., '(Chicago)', '(Detroit)')
    - Removes the word "County" (election data typically excludes it)
    - Strips and normalizes whitespace
    
    Examples:
    'Philadelphia, PA' → 'Philadelphia'
    'Cook County (Chicago), IL' → 'Cook'
    'Wayne County (Detroit), MI' → 'Wayne'
    'Alameda County, CA' → 'Alameda'
    """
    if not isinstance(juris_str, str):
        return ''
    
    # Remove state code (everything after last comma)
    if ',' in juris_str:
        juris_str = juris_str.rsplit(',', 1)[0]
    
    # Remove parenthetical content
    juris_str = re.sub(r'\s*\([^)]*\)', '', juris_str)
    
    # Remove the word "County" (case-insensitive)
    juris_str = re.sub(r'\bCounty\b', '', juris_str, flags=re.IGNORECASE)
    
    # Clean up multiple spaces and strip
    juris_str = re.sub(r'\s+', ' ', juris_str).strip()
    
    return juris_str


NICKNAME_MAP = {
    'kim': 'kimberly',
    'kimberly': 'kim',
    'bob': 'robert',
    'robert': 'bob',
    'mike': 'michael',
    'michael': 'mike',
    'dan': 'daniel',
    'daniel': 'dan',
    'joe': 'joseph',
    'joseph': 'joe',
    'bill': 'william',
    'william': 'bill',
    'tom': 'thomas',
    'thomas': 'tom',
    'jim': 'james',
    'james': 'jim',
    'dave': 'david',
    'david': 'dave',
    'steve': 'steven',
    'steven': 'steve',
}


def norm_first(s: str) -> str:
    """Normalize first name: lowercase, strip, handle nicknames."""
    if not isinstance(s, str) or not s.strip():
        return ''
    s = s.lower().strip()
    return NICKNAME_MAP.get(s, s)


def first_name_compatible(a: str, b: str) -> bool:
    """
    Returns True if empty, exact, prefix, or same initial.
    Also normalizes nicknames (e.g., Kim↔Kimberly).
    """
    a = norm_first(a)
    b = norm_first(b)
    if not a or not b:
        return True
    if a == b:
        return True
    if a.startswith(b) or b.startswith(a):
        return True
    if a[0] == b[0]:
        return True
    return False


# ---------------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------------
class DataLoader:
    def __init__(self, survey_file: str, election_file: str):
        self.survey_file = Path(survey_file)
        self.election_file = Path(election_file)

    def load(self) -> "DataLoader":
        print('='*80) ; print('LOADING DATA') ; print('='*80)

        # Survey core
        try:
            self.df_survey = pd.read_csv(self.survey_file, skiprows=[1])
            self.questions_df = pd.read_csv(self.survey_file, nrows=1)
            print(f"✓ Loaded survey ({len(self.df_survey)} rows) with Qualtrics header")
        except Exception:
            self.df_survey = pd.read_csv(self.survey_file)
            self.questions_df = None
            print(f"✓ Loaded survey ({len(self.df_survey)} rows) in simple CSV format")

        # Consent filter
        if 'con' in self.df_survey.columns:
            before = len(self.df_survey)
            self.df_survey = self.df_survey[self.df_survey['con'] == 'Agree'].copy()
            print(f"✓ Consent filter: {before} → {len(self.df_survey)}")

        self.total_respondents = len(self.df_survey)

        # National completion gate
        if 'more' in self.df_survey.columns:
            self.completed_national_section = int(self.df_survey['more'].notna().sum())
        else:
            self.completed_national_section = self.total_respondents
        print(f"✓ Completed national section: {self.completed_national_section}")

        # Elections
        try:
            self.df_elections = pd.read_csv(self.election_file)
            print(f"✓ Loaded elections ({len(self.df_elections)} records)")
        except Exception as e:
            print(f"✗ Failed to load elections: {e}")
            self.df_elections = pd.DataFrame()

        # Build prosecutors
        self.build_prosecutor_list()
        return self

    # ------------------------------
    def build_prosecutor_list(self):
        print('Building prosecutor list…')

        # 1) Notables
        notable_cols = [c for c in self.df_survey.columns if c.startswith('notable_')]
        df_notable_melt = self.df_survey.melt(
            id_vars=['ResponseId','RespondentState'], value_vars=notable_cols,
            var_name='q_id', value_name='rating_str'
        )
        df_notable = df_notable_melt.groupby('q_id').agg(
            total_ratings=('rating_str','count'),
            substantive_ratings=('rating_str', lambda x: x.notna().sum())
        ).reset_index()
        df_notable['name'] = df_notable['q_id'].map(lambda x: Config.DA_NAMES[x][0])
        df_notable['jurisdiction'] = df_notable['q_id'].map(lambda x: Config.DA_NAMES[x][1])
        df_notable['is_notable'] = True
        df_notable['familiarity_rate'] = (df_notable['substantive_ratings'] / max(1,self.completed_national_section)) * 100

        # derive state name from trailing ", XX"
        df_notable['state_name'] = df_notable['jurisdiction'].apply(state_from_jurisdiction)
        
        # NEW: Clean jurisdiction for matching (removes state codes and parentheticals)
        df_notable['jurisdiction_clean'] = df_notable['jurisdiction'].apply(clean_jurisdiction_for_matching)

        # split fn/ln (best effort for notables)
        def split_name(full: str) -> Tuple[Optional[str], Optional[str]]:
            if not isinstance(full, str) or not full.strip():
                return (None, None)
            parts = full.strip().split()
            if len(parts) == 1:
                return (parts[0], None)
            return (' '.join(parts[:-1]), parts[-1])
        fnln = df_notable['name'].apply(lambda x: pd.Series(split_name(x), index=['fname','lname']))
        df_notable = pd.concat([df_notable, fnln], axis=1)

        # 2) State items – columns that *start with* a US state name
        state_cols = [c for c in self.df_survey.columns if any(c.startswith(s + '_') for s in Config.US_STATE_NAMES)]
        state_melt = self.df_survey.melt(
            id_vars=['ResponseId','RespondentState'], value_vars=state_cols,
            var_name='q_id', value_name='rating_str'
        )
        state_melt['state'] = state_melt['q_id'].str.split('_').str[0]

        # Engaged in-state denominators: in-state respondents with ≥1 response
        st_resp = state_melt[state_melt['RespondentState'] == state_melt['state']]
        engaged = (st_resp.dropna(subset=['rating_str'])
                         .groupby('state')['ResponseId']
                         .nunique())
        self.state_denominators = engaged.to_dict()
        print(f"✓ State denominators computed for {len(self.state_denominators)} states")

        df_state = state_melt.groupby('q_id').agg(
            total_ratings=('rating_str','count'),
            substantive_ratings=('rating_str', lambda x: x.notna().sum())
        ).reset_index()
        df_state['state_name'] = df_state['q_id'].str.split('_').str[0]

        # Parse names from the first row (Qualtrics labels): often "… - Last\tFirst\tJurisdiction"
        def parse_name(q_id: str):
            if self.questions_df is None or q_id not in self.questions_df.columns:
                return pd.Series({'fname': None, 'lname': None, 'jurisdiction': None})
            qtext = str(self.questions_df[q_id].iloc[0])
            tail = qtext.split(' - ')[-1] if ' - ' in qtext else qtext
            parts = [p.strip() for p in tail.split('\t') if p.strip()]
            if len(parts) >= 3:
                lname, fname, juris = parts[0], parts[1], parts[2]
                return pd.Series({'fname': fname, 'lname': lname, 'jurisdiction': juris})
            elif len(parts) == 2:
                lname, fname = parts[0], parts[1]
                return pd.Series({'fname': fname, 'lname': lname, 'jurisdiction': None})
            return pd.Series({'fname': None, 'lname': None, 'jurisdiction': None})

        parsed = df_state['q_id'].apply(parse_name)
        df_state = pd.concat([df_state, parsed], axis=1)

        # Keep only real person items (must have both first & last names)
        df_state['fname'] = df_state['fname'].astype('string').str.strip()
        df_state['lname'] = df_state['lname'].astype('string').str.strip()
        before_state_rows = len(df_state)
        df_state = df_state[df_state['fname'].notna() & df_state['lname'].notna() & (df_state['fname'] != '') & (df_state['lname'] != '')]
        dropped_nonpersons = before_state_rows - len(df_state)
        print(f"✓ State items retained (named people): {len(df_state)}  | dropped non-person items: {dropped_nonpersons}")

        df_state['name'] = (df_state['fname'] + ' ' + df_state['lname']).str.strip()
        df_state['is_notable'] = False
        df_state['familiarity_rate'] = (df_state['substantive_ratings'] / df_state['state_name'].map(self.state_denominators)) * 100
        
        # State prosecutors already have clean jurisdictions (just district names)
        df_state['jurisdiction_clean'] = df_state['jurisdiction']

        # Combine
        keep_cols = ['q_id','name','fname','lname','state_name','jurisdiction','jurisdiction_clean','is_notable','substantive_ratings','familiarity_rate']
        self.df_prosecutors = pd.concat([df_notable[keep_cols], df_state[keep_cols]], ignore_index=True)

        # Ideology means (1–4), ignoring Not Familiar
        rating_map = Config.RATING_MAP
        all_melt = pd.concat([df_notable_melt, state_melt], ignore_index=True)
        all_melt['rating_val'] = all_melt['rating_str'].map(rating_map)
        ideology_scores = all_melt.groupby('q_id')['rating_val'].agg(['mean','std','count']).reset_index()
        ideology_scores.rename(columns={'mean':'mean_score','std':'sd_score','count':'n_scores'}, inplace=True)
        self.df_prosecutors = self.df_prosecutors.merge(ideology_scores, on='q_id', how='left')

        # Clean
        self.df_prosecutors.dropna(subset=['name'], inplace=True)
        print(f"✓ Master list built: {len(self.df_prosecutors)} prosecutors")

# ---------------------------------------------------------------------------------
# Analyzer (including robust matching)
# ---------------------------------------------------------------------------------
class ProsecutorAnalyzer:
    def __init__(self, loader: DataLoader):
        self.loader = loader
        self.df_survey = loader.df_survey
        self.df_elections = loader.df_elections
        self.df_prosecutors = loader.df_prosecutors.copy()

    # ------------------------------
    def build_prosecutor_dataset(self):
        print('='*80) ; print('BUILDING DATASET') ; print('='*80)
        self.df_prosecutors_filtered = self.df_prosecutors[self.df_prosecutors['substantive_ratings'] >= Config.MIN_RATINGS_THRESHOLD].copy()
        print(f"✓ Total prosecutors: {len(self.df_prosecutors)}")
        print(f"✓ Filtered (≥{Config.MIN_RATINGS_THRESHOLD} ratings): {len(self.df_prosecutors_filtered)}")

    # ------------------------------
    def _std(self, s: pd.Series) -> pd.Series:
        return std_series(s)

    def _prepare_matching_frames(self):
        df_s = self.df_prosecutors.copy()
        df_s['std_state'] = self._std(df_s['state_name'])
        # Use jurisdiction_clean instead of jurisdiction for matching
        df_s['std_dist'] = self._std(df_s['jurisdiction_clean'])
        df_s['std_fname'] = self._std(df_s['fname'])
        df_s['std_lname'] = self._std(df_s['lname'])

        df_e = self.df_elections.copy()
        for c_old, c_new in [('state','std_state'),('district','std_dist'),('cand_fname','std_fname'),('cand_lname','std_lname')]:
            df_e[c_new] = std_series(df_e[c_old])
        return df_s, df_e

    # ------------------------------
    def _hierarchical_match(self, df_s: pd.DataFrame, df_e: pd.DataFrame) -> pd.DataFrame:
        # Group elections by (state, dist, lname)
        grp = df_e.groupby(['std_state','std_dist','std_lname'])

        methods = []
        matched_rows = []

        # For strict 4-key, precompute a unique table favoring winners
        strict4 = (df_e.sort_values(['winner_general','incum_chall','election_year'], ascending=[False, False, False])
                      .drop_duplicates(['std_state','std_dist','std_fname','std_lname']))
        strict4_idx = strict4.set_index(['std_state','std_dist','std_fname','std_lname'])

        for _, r in df_s.iterrows():
            key4 = (r['std_state'], r['std_dist'], r['std_fname'], r['std_lname'])
            key3 = (r['std_state'], r['std_dist'], r['std_lname'])

            # Step A: strict 4-key
            m = None
            if key4 in strict4_idx.index:
                m = strict4_idx.loc[key4]
                method = 'strict_4'
                first_ok = True
            else:
                # Step B/C: candidates on 3-key
                if key3 in grp.groups:
                    cand = grp.get_group(key3).copy()
                    # B: first-name compatibility filter
                    cand['first_ok'] = cand['cand_fname'].apply(lambda x: first_name_compatible(r.get('fname',''), x))
                    cand1 = cand[cand['first_ok']]
                    if len(cand1) > 0:
                        m = (cand1.sort_values(['winner_general','incum_chall','election_year'], ascending=[False, False, False])
                                   .iloc[0])
                        method = 'relaxed_3_fname_compat'
                        first_ok = True
                    else:
                        # C: fallback 3-key without first-name guard
                        m = (cand.sort_values(['winner_general','incum_chall','election_year'], ascending=[False, False, False])
                                 .iloc[0])
                        method = 'fallback_3'
                        first_ok = False
                else:
                    method = 'no_match'

            methods.append(method)
            if m is not None and method != 'no_match':
                mdict = m.to_dict()
            else:
                mdict = {k: np.nan for k in df_e.columns}
            mdict['match_method'] = method
            mdict['first_name_compatible'] = np.nan if method in ('strict_4','no_match') else first_ok
            matched_rows.append(mdict)

        df_match = pd.DataFrame(matched_rows)
        out = pd.concat([df_s.reset_index(drop=True), df_match.reset_index(drop=True)], axis=1)
        return out

    # ------------------------------
    def match_with_elections(self):
        print('='*80) ; print('MATCHING WITH ELECTIONS') ; print('='*80)
        if self.df_elections.empty:
            print('✗ Elections frame empty – skipping.')
            self.df_matched_all = pd.DataFrame()
            self.df_matched = pd.DataFrame()
            return

        df_s, df_e = self._prepare_matching_frames()
        self.df_matched_all = self._hierarchical_match(df_s, df_e)
        self.df_matched = self.df_matched_all[self.df_matched_all['substantive_ratings'] >= Config.MIN_RATINGS_THRESHOLD].copy()

        n_all = len(self.df_matched_all)
        hits_all = int(self.df_matched_all['election_year'].notna().sum())
        n_f = len(self.df_matched)
        hits_f = int(self.df_matched['election_year'].notna().sum())
        print(f"✓ Matched (ALL): {hits_all}/{n_all} ({hits_all/n_all*100:.1f}%)")
        print(f"✓ Matched (FILTERED ≥{Config.MIN_RATINGS_THRESHOLD}): {hits_f}/{n_f} ({(hits_f/max(1,n_f))*100:.1f}%)")

    # ------------------------------ Optional quick looks -------------------------
    def analyze_familiarity(self):
        if not _HAVE_SEABORN:
            print('[viz] seaborn not installed – skipping familiarity plot')
            return
        import seaborn as sns  # local import for safety
        fig, ax = plt.subplots(figsize=(7,4))
        sns.boxplot(data=self.df_prosecutors, x='is_notable', y='familiarity_rate', ax=ax)
        ax.set_xticklabels(['State-Level','National Notable'])
        ax.set_title(f'Familiarity by Notability (N={len(self.df_prosecutors)})')
        ax.set_ylabel('Familiarity (%)')
        ax.set_xlabel('')
        plt.tight_layout()
        plt.savefig(Config.VIZ_DIR / 'familiarity_by_notability.png', dpi=200)
        plt.close()
        print('✓ Saved familiarity_by_notability.png')

    def run_models(self):
        if not _HAVE_SM:
            print('[models] statsmodels not installed – skipping OLS models')
            return
        import statsmodels.formula.api as smf  # local
        df = self.df_matched_all.copy()
        df['is_winner'] = (df['winner_general'] == 'W').astype(int)
        df['is_contested'] = (df['general_contested_reconciled'] == 'contested').astype(int)
        df['is_incumbent'] = (df['incum_chall'] == 'I').astype(int)
        df['vote_share'] = pd.to_numeric(df['vote_percent_general'], errors='coerce')

        mdat = df.dropna(subset=['mean_score'])
        if len(mdat) > 10:
            try:
                model = smf.ols('mean_score ~ is_winner + is_contested + is_incumbent + familiarity_rate', data=mdat).fit()
                print(model.summary())
            except Exception as e:
                print('[models] OLS failed:', e)

    # ------------------------------
    def export(self):
        print('='*80) ; print('EXPORTS') ; print('='*80)
        ensure_dir(Config.OUTPUT_DIR)
        ensure_dir(Config.VIZ_DIR)

        p_all = Config.OUTPUT_DIR / 'all_prosecutors_CORRECTED.csv'
        p_filt = Config.OUTPUT_DIR / 'prosecutors_filtered_ideology_CORRECTED.csv'
        p_match_all = Config.OUTPUT_DIR / 'matched_prosecutors_FULL_CORRECTED.csv'
        p_match_f = Config.OUTPUT_DIR / 'matched_prosecutors_FILTERED_CORRECTED.csv'
        p_audit = Config.OUTPUT_DIR / 'matched_prosecutors_AUDIT.csv'
        p_sum = Config.OUTPUT_DIR / 'CORRECTED_FINDINGS_SUMMARY.txt'

        self.df_prosecutors.to_csv(p_all, index=False)
        self.df_prosecutors_filtered.to_csv(p_filt, index=False)
        self.df_matched_all.to_csv(p_match_all, index=False)
        self.df_matched.to_csv(p_match_f, index=False)

        # audit file (subset of key columns)
        audit_cols = [
            'name','fname','lname','state_name','jurisdiction','jurisdiction_clean','is_notable','substantive_ratings','familiarity_rate',
            'cand_fname','cand_lname','state','district','election_year','winner_general','incum_chall',
            'match_method','first_name_compatible'
        ]
        keep = [c for c in audit_cols if c in self.df_matched_all.columns]
        self.df_matched_all[keep].to_csv(p_audit, index=False)

        # summary
        lines = []
        lines.append('CORRECTED ANALYSIS FINDINGS')
        lines.append('===========================')
        lines.append(f"Date: {pd.Timestamp.now():%Y-%m-%d %H:%M}")
        lines.append('')
        lines.append(f"Respondents (after consent): {self.loader.total_respondents}")
        lines.append(f"Completed national section: {self.loader.completed_national_section}")
        lines.append(f"States with engaged respondents: {len(self.loader.state_denominators)}")
        lines.append('')
        lines.append(f"All prosecutors: {len(self.df_prosecutors)}  | Notables: {int(self.df_prosecutors['is_notable'].sum())}  | State: {int((~self.df_prosecutors['is_notable']).sum())}")
        lines.append(f"Filtered (≥{Config.MIN_RATINGS_THRESHOLD} ratings): {len(self.df_prosecutors_filtered)}")
        lines.append('')
        hits_all = int(self.df_matched_all['election_year'].notna().sum())
        hits_f = int(self.df_matched['election_year'].notna().sum())
        lines.append(f"Matched (all prosecutors): {hits_all}/{len(self.df_matched_all)} ({hits_all/max(1,len(self.df_matched_all))*100:.1f}%)")
        lines.append(f"Matched (filtered): {hits_f}/{len(self.df_matched)} ({hits_f/max(1,len(self.df_matched))*100:.1f}%)")
        lines.append('')
        
        # Breakdown by notable vs state
        notable_mask = self.df_matched_all['is_notable'] == True
        notable_matched = int(self.df_matched_all[notable_mask]['election_year'].notna().sum())
        notable_total = int(notable_mask.sum())
        state_matched = int(self.df_matched_all[~notable_mask]['election_year'].notna().sum())
        state_total = int((~notable_mask).sum())
        
        lines.append('MATCHING BREAKDOWN:')
        lines.append(f"Notable prosecutors matched: {notable_matched}/{notable_total} ({notable_matched/max(1,notable_total)*100:.1f}%)")
        lines.append(f"State prosecutors matched: {state_matched}/{state_total} ({state_matched/max(1,state_total)*100:.1f}%)")

        Path(p_sum).write_text('\n'.join(lines), encoding='utf-8')
        print('✓ All exports complete.')
        print(f"   {p_all}")
        print(f"   {p_filt}")
        print(f"   {p_match_all}")
        print(f"   {p_match_f}")
        print(f"   {p_audit}")
        print(f"   {p_sum}")

# ---------------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------------
def main():
    loader = DataLoader(Config.SURVEY_FILE, Config.ELECTION_FILE)
    loader.load()

    analyzer = ProsecutorAnalyzer(loader)
    analyzer.build_prosecutor_dataset()
    analyzer.match_with_elections()
    analyzer.analyze_familiarity()
    analyzer.run_models()
    analyzer.export()

    print('='*80)
    print('✓ SCRIPT COMPLETE')
    print('='*80)


if __name__ == '__main__':
    main()

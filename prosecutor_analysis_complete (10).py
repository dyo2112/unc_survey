#!/usr/bin/env python3
"""
================================================================================
PROSECUTOR IDEOLOGY & FAMILIARITY – COMPREHENSIVE MASTER (2025-10-24)
================================================================================

This script includes ALL analyses from the comprehensive report.
Original data loading/matching preserved - ONLY analysis methods added.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
import matplotlib.pyplot as plt
from scipy import stats

# Optional imports
try:
    import seaborn as sns
    _HAVE_SEABORN = True
except ImportError:
    _HAVE_SEABORN = False

try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    _HAVE_SM = True
except ImportError:
    _HAVE_SM = False


# ---------------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------------
class Config:
    SURVEY_FILE = 'PP+survey+draft_October+16,+2025_13.32.csv'
    ELECTION_FILE = 'elections_with_reconciled_contested.csv'
    OUTPUT_DIR = Path('outputs/corrected')
    VIZ_DIR = OUTPUT_DIR / 'visualizations'
    MIN_RATINGS_THRESHOLD = 10

    RATING_MAP = {
        'Very Progressive': 4,
        'Progressive': 3,
        'Traditional': 2,
        'Very Traditional': 1
    }

    NOTABLE_PROSECUTORS = {
        'notable_1': ('Larry Krasner', 'Philadelphia, PA'),
        'notable_2': ('Kim Gardner', 'St. Louis, MO'),
        'notable_3': ('Marilyn Mosby', 'Baltimore, MD'),
        'notable_4': ('George Gascón', 'Los Angeles County, CA'),
        'notable_5': ('Chesa Boudin', 'San Francisco, CA'),
        'notable_6': ('Brooke Jenkins', 'San Francisco, CA'),
        'notable_7': ('Alvin Bragg', 'Manhattan, NY'),
        'notable_8': ('Kim Foxx', 'Cook County (Chicago), IL'),
        'notable_9': ('Rachael Rollins', 'Suffolk County (Boston), MA'),
        'notable_10': ('Cyrus Vance Jr.', 'Manhattan, NY'),
        'notable_11': ('John Creuzot', 'Dallas County, TX'),
        'notable_12': ('Joe Gonzales', 'Bexar County (San Antonio), TX'),
        'notable_13': ('José Garza', 'Travis County (Austin), TX'),
        'notable_14': ('Kimberly Graham', 'Travis County (Austin), TX'),
        'notable_15': ('Wesley Bell', 'St. Louis County, MO'),
        'notable_16': ('Aramis Ayala', 'Orange County, FL'),
        'notable_17': ('Andrew Warren', 'Hillsborough County (Tampa), FL'),
        'notable_18': ('Monique Worrell', 'Orange County, FL'),
        'notable_19': ('Buta Biberaj', 'Loudoun County, VA'),
        'notable_20': ('Steve Descano', 'Fairfax County, VA'),
        'notable_21': ('Parisa Dehghani-Tafti', 'Arlington County, VA'),
        'notable_22': ('Diana Becton', 'Contra Costa County, CA'),
        'notable_23': ('Pamela Price', 'Alameda County, CA'),
        'notable_24': ('Jeff Rosen', 'Santa Clara County, CA'),
        'notable_25': ('Mike Schmidt', 'Multnomah County (Portland), OR'),
        'notable_26': ('Nathan Vasquez', 'Multnomah County (Portland), OR'),
        'notable_27': ('Keith Kaneshiro', 'Honolulu, HI'),
        'notable_28': ('Kari Brandenburg', 'Bernalillo County (Albuquerque), NM'),
        'notable_29': ('Raúl Torrez', 'Bernalillo County (Albuquerque), NM'),
        'notable_30': ('Mary Moriarty', 'Hennepin County (Minneapolis), MN'),
        'notable_31': ('John Choi', 'Ramsey County (St. Paul), MN'),
        'notable_32': ('Eli Savit', 'Washtenaw County (Ann Arbor), MI'),
        'notable_33': ('Karen McDonald', 'Oakland County, MI'),
        'notable_34': ('Kym Worthy', 'Wayne County (Detroit), MI'),
        'notable_35': ('Rod Underhill', 'Multnomah County (Portland), OR'),
        'notable_36': ('Rachel Mitchell', 'Maricopa County (Phoenix), AZ'),
        'notable_37': ('Allison Emery', 'Maricopa County (Phoenix), AZ'),
        'notable_38': ('Mike Schmidt', 'Multnomah County (Portland), OR'),
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
    'WI': 'Wisconsin', 'WY': 'Wyoming'
}


NICKNAME_MAP = {
    'kim': 'kimberly', 'kimberly': 'kim', 'bob': 'robert', 'robert': 'bob',
    'mike': 'michael', 'michael': 'mike', 'dan': 'daniel', 'daniel': 'dan',
    'joe': 'joseph', 'joseph': 'joe', 'bill': 'william', 'william': 'bill',
    'tom': 'thomas', 'thomas': 'tom', 'jim': 'james', 'james': 'jim',
    'dave': 'david', 'david': 'dave', 'steve': 'steven', 'steven': 'steve',
}


# ---------------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def std_series(s: pd.Series) -> pd.Series:
    """Standardize text: lowercase, strip, collapse whitespace."""
    return s.astype(str).str.lower().str.strip().str.replace(r'\s+', ' ', regex=True)


def extract_state_from_jurisdiction(juris_str: str) -> str:
    """Extract state name from jurisdiction string."""
    if not isinstance(juris_str, str):
        return None
    parts = [p.strip() for p in juris_str.split(',')]
    if len(parts) >= 2:
        state_abbrev = parts[-1].strip().upper()
        return STATE_ABBREV_TO_NAME.get(state_abbrev, state_abbrev)
    return None


def clean_jurisdiction_for_matching(juris_str: str) -> str:
    """Clean jurisdiction string for matching with election data."""
    if not isinstance(juris_str, str):
        return ''
    
    if ',' in juris_str:
        juris_str = juris_str.rsplit(',', 1)[0]
    
    juris_str = re.sub(r'\s*\([^)]*\)', '', juris_str)
    juris_str = re.sub(r'\bCounty\b', '', juris_str, flags=re.IGNORECASE)
    juris_str = re.sub(r'\s+', ' ', juris_str).strip()
    
    return juris_str


def norm_first(s: str) -> str:
    """Normalize first name: lowercase, strip, handle nicknames."""
    if not isinstance(s, str) or not s.strip():
        return ''
    s = s.lower().strip()
    return NICKNAME_MAP.get(s, s)


def first_name_compatible(a: str, b: str) -> bool:
    """Returns True if empty, exact, prefix, or same initial."""
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
# Loader - EXACT ORIGINAL VERSION
# ---------------------------------------------------------------------------------
class DataLoader:
    def __init__(self, survey_file: str, election_file: str):
        self.survey_file = Path(survey_file)
        self.election_file = Path(election_file)

    def load(self) -> "DataLoader":
        print('='*80)
        print('LOADING DATA')
        print('='*80)

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

        # Build prosecutors - ORIGINAL METHOD
        self._build_prosecutors_original()
        return self

    def _build_prosecutors_original(self):
        """CORRECTED familiarity calculation - only 1-4 ratings count as familiar."""
        print('Building prosecutor list…')
        
        # Check for required columns
        if 'ResponseId' not in self.df_survey.columns:
            self.df_survey['ResponseId'] = range(len(self.df_survey))
        
        # Q1 contains the respondent's state
        if 'Q1' in self.df_survey.columns:
            self.df_survey['RespondentState'] = self.df_survey['Q1']
        elif 'RespondentState' not in self.df_survey.columns:
            if 'state' in self.df_survey.columns:
                self.df_survey['RespondentState'] = self.df_survey['state']
            else:
                self.df_survey['RespondentState'] = None
        
        # 1) NOTABLE PROSECUTORS
        notable_cols = [c for c in self.df_survey.columns if c.startswith('notable_')]
        df_notable_melt = self.df_survey.melt(
            id_vars=['ResponseId', 'RespondentState'],
            value_vars=notable_cols,
            var_name='q_id', 
            value_name='rating_str'
        )
        
        # CRITICAL: Substantive ratings = ONLY ratings 1-4, NOT "Not Familiar" or blanks
        rating_values = list(Config.RATING_MAP.keys())  # ['Very Progressive', 'Progressive', 'Traditional', 'Very Traditional']
        
        df_notable = df_notable_melt.groupby('q_id').agg(
            total_ratings=('rating_str', 'count'),
            substantive_ratings=('rating_str', lambda x: x.isin(rating_values).sum())
        ).reset_index()
        
        df_notable['name'] = df_notable['q_id'].map(lambda x: Config.NOTABLE_PROSECUTORS[x][0])
        df_notable['jurisdiction'] = df_notable['q_id'].map(lambda x: Config.NOTABLE_PROSECUTORS[x][1])
        df_notable['is_notable'] = True
        
        # Familiarity = substantive_ratings / all who completed national section (407)
        df_notable['familiarity_rate'] = (df_notable['substantive_ratings'] / max(1, self.completed_national_section)) * 100

        df_notable['state_name'] = df_notable['jurisdiction'].apply(extract_state_from_jurisdiction)
        df_notable['jurisdiction_clean'] = df_notable['jurisdiction'].apply(clean_jurisdiction_for_matching)

        def split_name(full: str):
            if not isinstance(full, str) or not full.strip():
                return (None, None)
            parts = full.strip().split()
            if len(parts) == 1:
                return (parts[0], None)
            return (' '.join(parts[:-1]), parts[-1])
        
        fnln = df_notable['name'].apply(lambda x: pd.Series(split_name(x), index=['fname','lname']))
        df_notable = pd.concat([df_notable, fnln], axis=1)

        # 2) STATE PROSECUTORS
        state_cols = [c for c in self.df_survey.columns 
                      if any(c.startswith(s + '_') for s in Config.US_STATE_NAMES)]
        
        state_melt = self.df_survey.melt(
            id_vars=['ResponseId', 'RespondentState'],
            value_vars=state_cols,
            var_name='q_id',
            value_name='rating_str'
        )
        state_melt['state'] = state_melt['q_id'].str.split('_').str[0]

        # CRITICAL: Denominators = ALL respondents who indicated that state in Q1
        # NOT just those who provided ratings
        st_resp_all = state_melt[state_melt['RespondentState'] == state_melt['state']]
        all_in_state = st_resp_all.groupby('state')['ResponseId'].nunique()
        self.state_denominators = all_in_state.to_dict()
        print(f"✓ State denominators computed for {len(self.state_denominators)} states")

        # Substantive ratings = ONLY ratings 1-4
        df_state = state_melt.groupby('q_id').agg(
            total_ratings=('rating_str', 'count'),
            substantive_ratings=('rating_str', lambda x: x.isin(rating_values).sum())
        ).reset_index()
        df_state['state_name'] = df_state['q_id'].str.split('_').str[0]

        # Parse names from Qualtrics labels in questions_df
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

        # Keep only real person items
        df_state['fname'] = df_state['fname'].astype('string').str.strip()
        df_state['lname'] = df_state['lname'].astype('string').str.strip()
        before_state_rows = len(df_state)
        df_state = df_state[df_state['fname'].notna() & df_state['lname'].notna() & 
                            (df_state['fname'] != '') & (df_state['lname'] != '')]
        dropped_nonpersons = before_state_rows - len(df_state)
        print(f"✓ State items retained (named people): {len(df_state)}  | dropped non-person items: {dropped_nonpersons}")

        df_state['name'] = (df_state['fname'] + ' ' + df_state['lname']).str.strip()
        df_state['is_notable'] = False
        
        # Familiarity = substantive_ratings / all in-state respondents who indicated that state
        df_state['familiarity_rate'] = (df_state['substantive_ratings'] / 
                                         df_state['state_name'].map(self.state_denominators)) * 100
        df_state['jurisdiction_clean'] = df_state['jurisdiction']

        # Combine
        keep_cols = ['q_id','name','fname','lname','state_name','jurisdiction',
                     'jurisdiction_clean','is_notable','substantive_ratings','familiarity_rate']
        self.df_prosecutors = pd.concat([df_notable[keep_cols], df_state[keep_cols]], ignore_index=True)

        # Ideology means (only from substantive ratings 1-4, not "Not Familiar")
        rating_map = Config.RATING_MAP
        all_melt = pd.concat([df_notable_melt, state_melt], ignore_index=True)
        all_melt['rating_val'] = all_melt['rating_str'].map(rating_map)
        ideology_scores = all_melt.groupby('q_id')['rating_val'].agg(['mean','std','count']).reset_index()
        ideology_scores.rename(columns={'mean':'mean_score','std':'sd_score','count':'n_scores'}, inplace=True)
        self.df_prosecutors = self.df_prosecutors.merge(ideology_scores, on='q_id', how='left')

        self.df_prosecutors.dropna(subset=['name'], inplace=True)
        print(f"✓ Master list built: {len(self.df_prosecutors)} prosecutors")


# ---------------------------------------------------------------------------------
# Analyzer - ORIGINAL MATCHING + NEW COMPREHENSIVE ANALYSES
# ---------------------------------------------------------------------------------
class ProsecutorAnalyzer:
    def __init__(self, loader: DataLoader):
        self.loader = loader
        self.df_survey = loader.df_survey
        self.df_elections = loader.df_elections
        self.df_prosecutors = loader.df_prosecutors.copy()

    def build_prosecutor_dataset(self):
        print('='*80)
        print('BUILDING DATASET')
        print('='*80)
        self.df_prosecutors_filtered = self.df_prosecutors[
            self.df_prosecutors['substantive_ratings'] >= Config.MIN_RATINGS_THRESHOLD
        ].copy()
        print(f"✓ Total prosecutors: {len(self.df_prosecutors)}")
        print(f"✓ Filtered (≥{Config.MIN_RATINGS_THRESHOLD} ratings): {len(self.df_prosecutors_filtered)}")

    def _std(self, s: pd.Series) -> pd.Series:
        return std_series(s)

    def _prepare_matching_frames(self):
        df_s = self.df_prosecutors.copy()
        df_s['std_state'] = self._std(df_s['state_name'])
        df_s['std_dist'] = self._std(df_s['jurisdiction_clean'])
        df_s['std_fname'] = self._std(df_s['fname'])
        df_s['std_lname'] = self._std(df_s['lname'])

        df_e = self.df_elections.copy()
        for c_old, c_new in [('state','std_state'),('district','std_dist'),
                              ('cand_fname','std_fname'),('cand_lname','std_lname')]:
            df_e[c_new] = std_series(df_e[c_old])
        return df_s, df_e

    def _hierarchical_match(self, df_s: pd.DataFrame, df_e: pd.DataFrame) -> pd.DataFrame:
        """ORIGINAL hierarchical matching algorithm."""
        grp = df_e.groupby(['std_state','std_dist','std_lname'])
        
        methods = []
        matched_rows = []
        
        strict4 = (df_e.sort_values(['winner_general','incum_chall','election_year'], 
                                    ascending=[False, False, False])
                      .drop_duplicates(['std_state','std_dist','std_fname','std_lname']))
        strict4_idx = strict4.set_index(['std_state','std_dist','std_fname','std_lname'])
        
        for _, r in df_s.iterrows():
            key4 = (r['std_state'], r['std_dist'], r['std_fname'], r['std_lname'])
            key3 = (r['std_state'], r['std_dist'], r['std_lname'])
            
            method = 'NONE'
            row_out = r.to_dict()
            
            if key4 in strict4_idx.index:
                matched = strict4_idx.loc[key4]
                method = 'STRICT_4KEY'
            elif key3 in grp.groups:
                cands = grp.get_group(key3)
                compat = cands[cands.apply(lambda x: first_name_compatible(r['std_fname'], x['std_fname']), axis=1)]
                if len(compat) > 0:
                    matched = (compat.sort_values(['winner_general','incum_chall','election_year'],
                                                  ascending=[False, False, False]).iloc[0])
                    method = 'RELAXED_3KEY'
                else:
                    matched = None
            else:
                matched = None
            
            if matched is not None:
                if isinstance(matched, pd.DataFrame):
                    matched = matched.iloc[0]
                for col in df_e.columns:
                    if col not in row_out:
                        row_out[col] = matched[col]
            
            row_out['match_method'] = method
            row_out['first_name_compatible'] = 'Y' if method != 'NONE' else 'N'
            matched_rows.append(row_out)
        
        return pd.DataFrame(matched_rows)

    def match_with_elections(self):
        print('='*80)
        print('MATCHING WITH ELECTIONS')
        print('='*80)
        
        if len(self.df_elections) == 0:
            print("⚠ No election data available")
            self.df_matched_all = pd.DataFrame()
            self.df_matched = pd.DataFrame()
            return
        
        df_s, df_e = self._prepare_matching_frames()
        self.df_matched_all = self._hierarchical_match(df_s, df_e)
        self.df_matched = self.df_matched_all[
            self.df_matched_all['substantive_ratings'] >= Config.MIN_RATINGS_THRESHOLD
        ].copy()
        
        n_all = len(self.df_matched_all)
        hits_all = int(self.df_matched_all['election_year'].notna().sum())
        n_f = len(self.df_matched)
        hits_f = int(self.df_matched['election_year'].notna().sum())
        print(f"✓ Matched (ALL): {hits_all}/{n_all} ({hits_all/n_all*100:.1f}%)")
        print(f"✓ Matched (FILTERED ≥{Config.MIN_RATINGS_THRESHOLD}): {hits_f}/{n_f} ({hits_f/max(1,n_f)*100:.1f}%)")

    # ========== NEW: COMPREHENSIVE ANALYSES FROM REPORT ==========
    
    def diagnose_prosecutor_ratings(self, prosecutor_name: str):
        """Diagnostic method to check raw ratings for a specific prosecutor."""
        print(f"\n{'='*80}")
        print(f"DIAGNOSTIC: Raw Ratings for {prosecutor_name}")
        print(f"{'='*80}")
        
        # Find in notable prosecutors first
        q_id = None
        for k, v in Config.NOTABLE_PROSECUTORS.items():
            if v[0] == prosecutor_name:
                q_id = k
                break
        
        if q_id and q_id in self.df_survey.columns:
            # Get raw ratings
            ratings = self.df_survey[q_id].value_counts().sort_index()
            print(f"\nRaw Rating Distribution for {prosecutor_name} ({q_id}):")
            for rating, count in ratings.items():
                print(f"  {rating}: {count} ({count/len(self.df_survey)*100:.1f}%)")
            
            # Calculate ideology score
            rating_map = Config.RATING_MAP
            numeric_ratings = self.df_survey[q_id].map(rating_map).dropna()
            if len(numeric_ratings) > 0:
                print(f"\nIdeology Statistics:")
                print(f"  Mean: {numeric_ratings.mean():.2f}")
                print(f"  Median: {numeric_ratings.median():.2f}")
                print(f"  Substantive ratings (1-4): {len(numeric_ratings)}")
                print(f"  Not Familiar: {(self.df_survey[q_id] == 'Not Familiar').sum()}")
        else:
            print(f"Could not find {prosecutor_name} in survey data")
    
    def run_all_analyses(self):
        """Run all analyses from the report.
        
        SAMPLE USAGE:
        - ALL PROSECUTORS: Familiarity comparisons (no filtering)
        - FULL MATCHED (df_matched_all): Electoral patterns, contestation, margins
        - FILTERED (df_matched, ≥10 ratings): Ideology comparisons, correlations, regression
        
        Per report: "Analyses of electoral patterns use the full matched sample of 191 
        prosecutors with election data, regardless of rating count. Analyses comparing 
        progressive vs. traditional prosecutors use only the filtered sample with ≥10 
        ratings (n=51) to ensure reliable ideology measures."
        """
        print('='*80)
        print('RUNNING COMPREHENSIVE ANALYSES')
        print('='*80)
        
        # Print sample sizes for transparency - COUNT UNIQUE PROSECUTORS
        print(f"\nSample sizes:")
        print(f"  All prosecutors: {len(self.df_prosecutors)}")
        print(f"  Filtered (≥{Config.MIN_RATINGS_THRESHOLD} ratings): {len(self.df_prosecutors_filtered)}")
        
        # Count UNIQUE prosecutors with election data (not rows)
        matched_unique = self.df_matched_all[self.df_matched_all['election_year'].notna()]['name'].nunique()
        filtered_matched_unique = self.df_matched[self.df_matched['election_year'].notna()]['name'].nunique()
        print(f"  Full matched (unique prosecutors with election data): {matched_unique}")
        print(f"  Filtered + matched (unique prosecutors): {filtered_matched_unique}")
        
        # Count UNIQUE prosecutors by notability AMONG MATCHED ONLY
        matched_only = self.df_matched_all[self.df_matched_all['election_year'].notna()]
        notable_matched_unique = matched_only[matched_only['is_notable'] == True]['name'].nunique()
        state_matched_unique = matched_only[matched_only['is_notable'] == False]['name'].nunique()
        print(f"    → Notable prosecutors with election data: {notable_matched_unique}")
        print(f"    → State prosecutors with election data: {state_matched_unique}")
        
        # Verify notable prosecutors
        total_notables = self.df_prosecutors[self.df_prosecutors['is_notable'] == True]['name'].nunique()
        print(f"\n  Verification:")
        print(f"    Total notable prosecutors: {total_notables} (should be 50)")
        print(f"    Notables matched: {notable_matched_unique}/{total_notables}")
        if notable_matched_unique < total_notables:
            unmatched_notables = self.df_prosecutors[
                (self.df_prosecutors['is_notable'] == True) & 
                (~self.df_prosecutors['name'].isin(matched_only[matched_only['is_notable'] == True]['name']))
            ]['name'].tolist()
            print(f"    Unmatched notables: {unmatched_notables}")
        print()
        
        self.results = {}
        
        # 1. Familiarity comparisons
        self._analyze_familiarity_comparisons()
        
        # 2. Electoral competition effects
        self._analyze_electoral_competition()
        
        # 3. Correlations
        self._analyze_correlations()
        
        # 4. Multivariate regression
        self._run_multivariate_regression()
        
        # 5. Close victory vulnerability
        self._analyze_close_victory_vulnerability()
        
        # 6. Contestation by ideology
        self._analyze_contestation_by_ideology()
        
        # 7. Descriptive statistics by groups
        self._compute_descriptive_statistics()
        
        self._print_results_summary()

    def _analyze_familiarity_comparisons(self):
        """Compare familiarity between notable and state prosecutors.
        SAMPLE: ALL PROSECUTORS (no election data or rating threshold required)"""
        print("\n" + "="*60)
        print("1. FAMILIARITY COMPARISONS")
        print("="*60)
        
        notable = self.df_prosecutors[self.df_prosecutors['is_notable'] == True]['familiarity_rate'].dropna()
        state = self.df_prosecutors[self.df_prosecutors['is_notable'] == False]['familiarity_rate'].dropna()
        
        # Calculate unfamiliarity
        notable_unfam = 100 - notable.mean()
        state_unfam = 100 - state.mean()
        
        # Check for sufficient data and variance
        if len(notable) > 1 and len(state) > 1 and notable.std() > 0 and state.std() > 0:
            t_stat, p_val = stats.ttest_ind(notable, state, equal_var=False)
        else:
            t_stat, p_val = np.nan, np.nan
            print(f"⚠ Warning: Insufficient variance for t-test")
            print(f"   Notable: n={len(notable)}, mean={notable.mean():.2f}, std={notable.std():.2f}")
            print(f"   State: n={len(state)}, mean={state.mean():.2f}, std={state.std():.2f}")
        
        print(f"Notable prosecutors - Mean familiarity: {notable.mean():.2f}%")
        print(f"  → Unfamiliarity rate: {notable_unfam:.2f}%")
        print(f"State prosecutors - Mean familiarity: {state.mean():.2f}%")
        print(f"  → Unfamiliarity rate: {state_unfam:.2f}%")
        print(f"Welch's t-test: t = {t_stat:.3f}, p = {p_val:.6f}")
        
        self.results['familiarity_comparison'] = {
            'notable_mean': notable.mean(),
            'notable_unfam': notable_unfam,
            'state_mean': state.mean(),
            'state_unfam': state_unfam,
            't_stat': t_stat,
            'p_value': p_val
        }

    def _analyze_electoral_competition(self):
        """Analyze effects of electoral competition on familiarity.
        SAMPLE: FULL MATCHED SAMPLE (all with election data, no rating threshold)"""
        print("\n" + "="*60)
        print("2. ELECTORAL COMPETITION EFFECTS")
        print("="*60)
        
        if 'general_contested_reconciled' not in self.df_matched_all.columns:
            print("⚠ Election contestation data not available")
            return
        
        df = self.df_matched_all[self.df_matched_all['general_contested_reconciled'].notna()].copy()
        
        contested = df[df['general_contested_reconciled'] == 'contested']['familiarity_rate'].dropna()
        uncontested = df[df['general_contested_reconciled'] == 'uncontested']['familiarity_rate'].dropna()
        
        # Overall comparison
        if len(contested) > 1 and len(uncontested) > 1 and contested.std() > 0 and uncontested.std() > 0:
            t_stat, p_val = stats.ttest_ind(contested, uncontested, equal_var=False)
        else:
            t_stat, p_val = np.nan, np.nan
            print(f"⚠ Warning: Insufficient variance for overall t-test")
            print(f"   Contested: n={len(contested)}, mean={contested.mean():.2f}, std={contested.std():.2f}")
            print(f"   Uncontested: n={len(uncontested)}, mean={uncontested.mean():.2f}, std={uncontested.std():.2f}")
        
        print(f"\nOverall (n={len(df)}):")
        print(f"  Contested (n={len(contested)}): {contested.mean():.2f}%")
        print(f"  Uncontested (n={len(uncontested)}): {uncontested.mean():.2f}%")
        print(f"  Difference: {contested.mean() - uncontested.mean():.2f} pp")
        print(f"  Welch's t = {t_stat:.3f}, p = {p_val:.4f}")
        
        print("\nBy Notability:")
        for notable_val, label in [(True, 'Notable'), (False, 'State')]:
            subset = df[df['is_notable'] == notable_val]
            if len(subset) > 10:
                contested_sub = subset[subset['general_contested_reconciled'] == 'contested']['familiarity_rate'].dropna()
                uncontested_sub = subset[subset['general_contested_reconciled'] == 'uncontested']['familiarity_rate'].dropna()
                
                if len(contested_sub) > 1 and len(uncontested_sub) > 1:
                    # Check variance
                    if contested_sub.std() > 0 and uncontested_sub.std() > 0:
                        t_stat_sub, p_val_sub = stats.ttest_ind(contested_sub, uncontested_sub, equal_var=False)
                    else:
                        t_stat_sub, p_val_sub = np.nan, np.nan
                        print(f"\n  {label} prosecutors - ⚠ Zero variance:")
                        print(f"    Contested: std={contested_sub.std():.4f}")
                        print(f"    Uncontested: std={uncontested_sub.std():.4f}")
                    
                    print(f"\n  {label} prosecutors (n={len(subset)}):")
                    print(f"    Contested (n={len(contested_sub)}): {contested_sub.mean():.2f}%")
                    print(f"    Uncontested (n={len(uncontested_sub)}): {uncontested_sub.mean():.2f}%")
                    print(f"    Difference: {contested_sub.mean() - uncontested_sub.mean():.2f} pp")
                    print(f"    Welch's t = {t_stat_sub:.3f}, p = {p_val_sub:.4f}")

    def _analyze_correlations(self):
        """Compute key correlations.
        SAMPLE: FILTERED (≥10 ratings) - requires reliable ideology scores"""
        print("\n" + "="*60)
        print("3. CORRELATION ANALYSES")
        print("="*60)
        
        df = self.df_matched[self.df_matched['mean_score'].notna()].copy()
        
        if len(df) > 10:
            # DIAGNOSTIC: Check ideology scale
            print(f"\nDIAGNOSTIC - Ideology Scale Check:")
            print(f"  Rating scale: 1=Very Traditional, 2=Traditional, 3=Progressive, 4=Very Progressive")
            print(f"  Mean ideology score: {df['mean_score'].mean():.2f}")
            print(f"  Min: {df['mean_score'].min():.2f}, Max: {df['mean_score'].max():.2f}")
            
            # Check specific high-profile prosecutors
            print(f"\n  HIGH-PROFILE PROSECUTOR CHECK:")
            high_profile = ['Chesa Boudin', 'Brooke Jenkins', 'Larry Krasner', 
                           'George Gascón', 'Rachel Mitchell', 'Kim Foxx']
            for name in high_profile:
                row = df[df['name'] == name]
                if len(row) > 0:
                    score = row.iloc[0]['mean_score']
                    fam = row.iloc[0]['familiarity_rate']
                    n_ratings = row.iloc[0]['n_scores'] if 'n_scores' in row.columns else 'N/A'
                    print(f"    {name}: score={score:.2f}, fam={fam:.1f}%, n_ratings={n_ratings}")
                    
                    # Flag suspicious cases
                    if name == 'Brooke Jenkins' and score > 3.0:
                        print(f"      ⚠ WARNING: Brooke Jenkins scored as Progressive ({score:.2f})")
                        print(f"         She replaced Chesa Boudin and should be Traditional!")
                    if name == 'Rachel Mitchell' and score > 2.5:
                        print(f"      ⚠ WARNING: Rachel Mitchell scored too high ({score:.2f})")
            
            print(f"\n  Sample of prosecutors:")
            sample_procs = df.nsmallest(3, 'mean_score')[['name', 'mean_score', 'familiarity_rate']]
            print("  Most Traditional:")
            for _, row in sample_procs.iterrows():
                print(f"    {row['name']}: score={row['mean_score']:.2f}, fam={row['familiarity_rate']:.1f}%")
            sample_procs = df.nlargest(3, 'mean_score')[['name', 'mean_score', 'familiarity_rate']]
            print("  Most Progressive:")
            for _, row in sample_procs.iterrows():
                print(f"    {row['name']}: score={row['mean_score']:.2f}, fam={row['familiarity_rate']:.1f}%")
            
            # Progressiveness-familiarity correlation
            r, p = stats.pearsonr(df['mean_score'], df['familiarity_rate'])
            print(f"\nProgressiveness-Familiarity Correlation:")
            print(f"  Pearson r = {r:.3f}, p = {p:.4f}")
            print(f"  n = {len(df)}")
            
            if r < 0:
                print(f"  ⚠ WARNING: NEGATIVE correlation detected!")
                print(f"     This suggests more progressive prosecutors have LOWER familiarity")
                print(f"     This contradicts the report (r=0.403, positive)")
                print(f"     Possible causes:")
                print(f"       1. Ideology scores are incorrectly coded (check Brooke Jenkins)")
                print(f"       2. Survey respondents misunderstood the scale")
                print(f"       3. The report had errors")
            
            self.results['progressiveness_familiarity_corr'] = {
                'r': r, 'p': p, 'n': len(df)
            }
            
            # Primary margin correlation (if available)
            if 'vote_percent_primary' in df.columns:
                primary_data = df[df['vote_percent_primary'].notna()].copy()
                if len(primary_data) > 10:
                    r_prim, p_prim = stats.pearsonr(primary_data['vote_percent_primary'], 
                                                     primary_data['familiarity_rate'])
                    print(f"\nPrimary Margin-Familiarity Correlation:")
                    print(f"  Pearson r = {r_prim:.3f}, p = {p_prim:.4f}")
                    print(f"  n = {len(primary_data)}")

    def _run_multivariate_regression(self):
        """Run multivariate OLS regression for familiarity.
        SAMPLE: FILTERED (≥10 ratings) - uses mean_score as predictor"""
        print("\n" + "="*60)
        print("4. MULTIVARIATE REGRESSION MODEL")
        print("="*60)
        
        if not _HAVE_SM:
            print("⚠ statsmodels not available - skipping regression")
            return
        
        df = self.df_matched.copy()
        
        df['is_contested'] = (df['general_contested_reconciled'] == 'contested').astype(int)
        df['is_incumbent'] = (df['incum_chall'] == 'I').astype(int)
        df['is_challenger'] = (df['incum_chall'] == 'C').astype(int)
        
        if 'election_year' in df.columns:
            election_counts = df.groupby('name')['election_year'].nunique().reset_index()
            election_counts.columns = ['name', 'num_elections']
            df = df.merge(election_counts, on='name', how='left')
        else:
            df['num_elections'] = 1
        
        # Prepare data for regression - one row per prosecutor
        reg_df = df[df['mean_score'].notna()].copy()
        reg_df = reg_df.drop_duplicates(subset='name')
        
        if len(reg_df) < 20:
            print(f"⚠ Insufficient data for regression (n={len(reg_df)})")
            return
        
        print(f"\nRegression sample (n={len(reg_df)}):")
        print(f"  Mean familiarity: {reg_df['familiarity_rate'].mean():.2f}%")
        print(f"  Mean ideology: {reg_df['mean_score'].mean():.2f}")
        print(f"  Contested: {reg_df['is_contested'].sum()} ({reg_df['is_contested'].mean()*100:.1f}%)")
        
        # Build formula
        formula = 'familiarity_rate ~ mean_score + is_contested + is_challenger + is_incumbent + num_elections + is_notable'
        
        try:
            model = smf.ols(formula, data=reg_df).fit()
            
            print("\nOLS Regression Results:")
            print(f"Dependent Variable: familiarity_rate")
            print(f"n = {int(model.nobs)}")
            print(f"R² = {model.rsquared:.3f}")
            print("\nCoefficients:")
            
            for var in ['mean_score', 'is_contested', 'is_challenger', 'is_incumbent', 
                       'num_elections', 'is_notable']:
                if var in model.params.index:
                    coef = model.params[var]
                    pval = model.pvalues[var]
                    sig = ""
                    if pval < 0.001:
                        sig = "***"
                    elif pval < 0.01:
                        sig = "**"
                    elif pval < 0.05:
                        sig = "*"
                    print(f"  {var:20s}: β = {coef:7.2f}  (p = {pval:.4f}) {sig}")
            
            # Check if coefficient signs match expectations
            if 'mean_score' in model.params.index:
                coef = model.params['mean_score']
                if coef < 0:
                    print(f"\n  ⚠ WARNING: mean_score coefficient is NEGATIVE ({coef:.2f})")
                    print(f"     This suggests progressive prosecutors have LOWER familiarity")
            
            self.results['regression'] = {
                'n': int(model.nobs),
                'r_squared': model.rsquared,
                'coefficients': dict(model.params),
                'pvalues': dict(model.pvalues)
            }
            
        except Exception as e:
            print(f"⚠ Regression failed: {e}")

    def _analyze_close_victory_vulnerability(self):
        """Analyze relationship between victory margin and recalls - USES FULL MATCHED SAMPLE."""
        print("\n" + "="*60)
        print("5. CLOSE VICTORY VULNERABILITY")
        print("="*60)
        
        recalled = ['Chesa Boudin', 'Pamela Price']
        
        # USE FULL MATCHED SAMPLE (all with election data, not just filtered)
        df = self.df_matched_all[self.df_matched_all['vote_percent_general'].notna()].copy()
        df['is_recalled'] = df['name'].isin(recalled)
        
        if df['is_recalled'].sum() > 0:
            recalled_margins = df[df['is_recalled']]['vote_percent_general']
            not_recalled_margins = df[~df['is_recalled']]['vote_percent_general']
            
            t_stat, p_val = stats.ttest_ind(recalled_margins, not_recalled_margins, equal_var=False)
            
            print(f"\nRecalled prosecutors (n={len(recalled_margins)}):")
            print(f"  Mean margin: {recalled_margins.mean():.1f}%")
            print(f"  Margins: {recalled_margins.tolist()}")
            
            print(f"\nNon-recalled prosecutors (n={len(not_recalled_margins)}):")
            print(f"  Mean margin: {not_recalled_margins.mean():.1f}%")
            print(f"  Median margin: {not_recalled_margins.median():.1f}%")
            
            print(f"\nWelch's t-test: t = {t_stat:.3f}, p = {p_val:.4f}")
            
            self.results['close_victory'] = {
                'recalled_mean': recalled_margins.mean(),
                'not_recalled_mean': not_recalled_margins.mean(),
                't_stat': t_stat,
                'p_value': p_val
            }
        else:
            print("\n⚠ No recalled prosecutors found in matched sample")

    def _analyze_contestation_by_ideology(self):
        """Analyze contestation rates by ideology.
        SAMPLE: FILTERED (≥10 ratings) - requires reliable ideology scores"""
        print("\n" + "="*60)
        print("6. CONTESTATION BY IDEOLOGY")
        print("="*60)
        
        df = self.df_matched[self.df_matched['mean_score'].notna()].copy()
        
        df['ideology_category'] = pd.cut(df['mean_score'], 
                                         bins=[0, 2, 3, 5],
                                         labels=['Traditional', 'Moderate', 'Progressive'])
        
        print("\nIdeology Distribution:")
        print(df['ideology_category'].value_counts())
        
        if 'general_contested_reconciled' in df.columns:
            print("\nGeneral Election Contestation Rates:")
            for cat in ['Traditional', 'Moderate', 'Progressive']:
                subset = df[df['ideology_category'] == cat]
                if len(subset) > 0:
                    contested_pct = (subset['general_contested_reconciled'] == 'contested').mean() * 100
                    print(f"  {cat:12s}: {contested_pct:5.1f}% (n={len(subset)})")
        
        if 'incum_chall' in df.columns:
            print("\nRan as Challenger:")
            for cat in ['Traditional', 'Moderate', 'Progressive']:
                subset = df[df['ideology_category'] == cat]
                if len(subset) > 0:
                    challenger_pct = (subset['incum_chall'] == 'C').mean() * 100
                    print(f"  {cat:12s}: {challenger_pct:5.1f}% (n={len(subset)})")

    def _compute_descriptive_statistics(self):
        """Compute descriptive statistics by groups.
        SAMPLE: FILTERED (≥10 ratings) - for ideology statistics"""
        print("\n" + "="*60)
        print("7. DESCRIPTIVE STATISTICS")
        print("="*60)
        
        df = self.df_prosecutors_filtered.copy()
        
        print(f"\nOverall (n={len(df)}):")
        print(f"  Mean familiarity: {df['familiarity_rate'].mean():.2f}%")
        print(f"  Median familiarity: {df['familiarity_rate'].median():.2f}%")
        print(f"  SD familiarity: {df['familiarity_rate'].std():.2f}%")
        
        if 'mean_score' in df.columns:
            ideology_df = df[df['mean_score'].notna()]
            print(f"\n  Mean ideology score: {ideology_df['mean_score'].mean():.2f}")
            print(f"  SD ideology score: {ideology_df['mean_score'].std():.2f}")
        
        print(f"\nBy Notability:")
        for notable_val, label in [(True, 'Notable'), (False, 'State')]:
            subset = df[df['is_notable'] == notable_val]
            print(f"\n  {label} (n={len(subset)}):")
            print(f"    Mean familiarity: {subset['familiarity_rate'].mean():.2f}%")
            print(f"    Median familiarity: {subset['familiarity_rate'].median():.2f}%")
            print(f"    SD familiarity: {subset['familiarity_rate'].std():.2f}%")

    def _print_results_summary(self):
        """Print comprehensive results summary with data quality checks."""
        print("\n" + "="*80)
        print("ANALYSIS RESULTS SUMMARY")
        print("="*80)
        
        # Data quality warnings
        warnings = []
        
        if 'familiarity_comparison' in self.results:
            fc = self.results['familiarity_comparison']
            print(f"\n✓ Familiarity Crisis:")
            print(f"  Notable unfamiliarity: {fc['notable_unfam']:.2f}%")
            print(f"  State unfamiliarity: {fc['state_unfam']:.2f}%")
            print(f"  Difference: {fc['state_unfam'] - fc['notable_unfam']:.2f} pp (p={fc['p_value']:.6f})")
            
            if np.isnan(fc['t_stat']):
                warnings.append("Familiarity comparison: t-test returned NaN (check variance)")
        
        if 'progressiveness_familiarity_corr' in self.results:
            pfc = self.results['progressiveness_familiarity_corr']
            print(f"\n✓ Progressiveness-Familiarity Correlation:")
            print(f"  r = {pfc['r']:.3f}, p = {pfc['p']:.4f}")
            
            if pfc['r'] < 0:
                warnings.append(f"CRITICAL: Progressiveness-familiarity correlation is NEGATIVE (r={pfc['r']:.3f})")
                warnings.append("  → This contradicts the report which shows positive correlation (r=0.403)")
                warnings.append("  → Possible causes: ideology scale inverted, familiarity calc error, or data change")
        
        if 'regression' in self.results:
            reg = self.results['regression']
            print(f"\n✓ Multivariate Regression:")
            print(f"  n = {reg['n']}, R² = {reg['r_squared']:.3f}")
            print(f"  Key predictors:")
            for var in ['mean_score', 'is_contested']:
                if var in reg['pvalues']:
                    sig = "*" if reg['pvalues'][var] < 0.05 else "ns"
                    print(f"    {var}: β={reg['coefficients'][var]:.2f}, p={reg['pvalues'][var]:.4f} {sig}")
            
            if 'mean_score' in reg['coefficients'] and reg['coefficients']['mean_score'] < 0:
                warnings.append(f"Regression: mean_score coefficient is negative ({reg['coefficients']['mean_score']:.2f})")
        
        if 'close_victory' in self.results:
            cv = self.results['close_victory']
            print(f"\n✓ Close Victory Vulnerability:")
            print(f"  Recalled mean margin: {cv['recalled_mean']:.1f}%")
            print(f"  Not recalled mean margin: {cv['not_recalled_mean']:.1f}%")
            print(f"  t = {cv['t_stat']:.3f}, p = {cv['p_value']:.4f}")
        
        # Print all warnings at the end
        if warnings:
            print("\n" + "="*80)
            print("⚠ DATA QUALITY WARNINGS")
            print("="*80)
            for i, warning in enumerate(warnings, 1):
                print(f"{i}. {warning}")

    # ========== ORIGINAL VISUALIZATION METHOD ==========
    
    def analyze_familiarity(self):
        """Original visualization method."""
        if not _HAVE_SEABORN:
            print('[viz] seaborn not installed – skipping familiarity plot')
            return
        import seaborn as sns
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
        """Original basic model method - kept for compatibility."""
        if not _HAVE_SM:
            print('[models] statsmodels not installed – skipping OLS models')
            return
        import statsmodels.formula.api as smf
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

    def export(self):
        """Export all results."""
        print('='*80)
        print('EXPORTS')
        print('='*80)
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

        # Audit file
        audit_cols = [
            'name','fname','lname','state_name','jurisdiction','jurisdiction_clean',
            'is_notable','substantive_ratings','familiarity_rate',
            'cand_fname','cand_lname','state','district','election_year',
            'winner_general','incum_chall','match_method','first_name_compatible'
        ]
        keep = [c for c in audit_cols if c in self.df_matched_all.columns]
        self.df_matched_all[keep].to_csv(p_audit, index=False)

        # Summary
        lines = []
        lines.append('COMPREHENSIVE ANALYSIS FINDINGS')
        lines.append('=' * 80)
        lines.append(f"Date: {pd.Timestamp.now():%Y-%m-%d %H:%M}")
        lines.append('')
        lines.append(f"Respondents (after consent): {self.loader.total_respondents}")
        lines.append(f"Completed national section: {self.loader.completed_national_section}")
        lines.append(f"States with engaged respondents: {len(self.loader.state_denominators)}")
        lines.append('')
        lines.append(f"All prosecutors: {len(self.df_prosecutors)}")
        lines.append(f"  Notable: {int(self.df_prosecutors['is_notable'].sum())}")
        lines.append(f"  State: {int((~self.df_prosecutors['is_notable']).sum())}")
        lines.append(f"Filtered (≥{Config.MIN_RATINGS_THRESHOLD} ratings): {len(self.df_prosecutors_filtered)}")
        lines.append('')
        
        hits_all = int(self.df_matched_all['election_year'].notna().sum())
        hits_f = int(self.df_matched['election_year'].notna().sum())
        lines.append(f"Matched (all): {hits_all}/{len(self.df_matched_all)} ({hits_all/max(1,len(self.df_matched_all))*100:.1f}%)")
        lines.append(f"Matched (filtered): {hits_f}/{len(self.df_matched)} ({hits_f/max(1,len(self.df_matched))*100:.1f}%)")
        lines.append('')
        
        notable_mask = self.df_matched_all['is_notable'] == True
        notable_matched = int(self.df_matched_all[notable_mask]['election_year'].notna().sum())
        notable_total = int(notable_mask.sum())
        state_matched = int(self.df_matched_all[~notable_mask]['election_year'].notna().sum())
        state_total = int((~notable_mask).sum())
        
        lines.append('MATCHING BREAKDOWN:')
        lines.append(f"Notable: {notable_matched}/{notable_total} ({notable_matched/max(1,notable_total)*100:.1f}%)")
        lines.append(f"State: {state_matched}/{state_total} ({state_matched/max(1,state_total)*100:.1f}%)")
        lines.append('')
        lines.append('=' * 80)
        lines.append('KEY FINDINGS:')
        lines.append('=' * 80)
        
        if hasattr(self, 'results'):
            if 'familiarity_comparison' in self.results:
                fc = self.results['familiarity_comparison']
                lines.append('')
                lines.append('1. FAMILIARITY CRISIS:')
                lines.append(f"   Notable unfamiliarity: {fc['notable_unfam']:.2f}%")
                lines.append(f"   State unfamiliarity: {fc['state_unfam']:.2f}%")
                lines.append(f"   t = {fc['t_stat']:.3f}, p = {fc['p_value']:.6f}")
            
            if 'progressiveness_familiarity_corr' in self.results:
                pfc = self.results['progressiveness_familiarity_corr']
                lines.append('')
                lines.append('2. PROGRESSIVENESS-FAMILIARITY CORRELATION:')
                lines.append(f"   r = {pfc['r']:.3f}, p = {pfc['p']:.4f}, n = {pfc['n']}")
            
            if 'regression' in self.results:
                reg = self.results['regression']
                lines.append('')
                lines.append('3. MULTIVARIATE REGRESSION:')
                lines.append(f"   n = {reg['n']}, R² = {reg['r_squared']:.3f}")
                for var in ['mean_score', 'is_contested', 'is_incumbent', 'num_elections']:
                    if var in reg['coefficients']:
                        lines.append(f"   {var}: β = {reg['coefficients'][var]:.2f}, p = {reg['pvalues'][var]:.4f}")
            
            if 'close_victory' in self.results:
                cv = self.results['close_victory']
                lines.append('')
                lines.append('4. CLOSE VICTORY VULNERABILITY:')
                lines.append(f"   Recalled margin: {cv['recalled_mean']:.1f}%")
                lines.append(f"   Not recalled margin: {cv['not_recalled_mean']:.1f}%")
                lines.append(f"   t = {cv['t_stat']:.3f}, p = {cv['p_value']:.4f}")

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
    
    # Run comprehensive analyses
    analyzer.run_all_analyses()
    
    # Original visualization
    analyzer.analyze_familiarity()
    
    # Original basic model (for backwards compatibility)
    analyzer.run_models()
    
    # Export results
    analyzer.export()

    print('='*80)
    print('✓ SCRIPT COMPLETE')
    print('='*80)


if __name__ == '__main__':
    main()

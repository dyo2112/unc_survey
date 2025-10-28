#!/usr/bin/env python3
"""
================================================================================
CORRECTED PROSECUTOR IDEOLOGY SURVEY - COMPREHENSIVE ANALYSIS
================================================================================

CRITICAL FIXES:
1. Notable prosecutor familiarity calculated correctly:
   - Denominator: 407 respondents who completed national section (answered 'more' question)
   - Formula: substantive_ratings / 407
   
2. State prosecutor familiarity calculated correctly:
   - Denominator: respondents from that state who engaged with state prosecutor questions
   - "Engaged" = responded to at least one state prosecutor from their state (not all blank)
   - Formula: substantive_ratings / state_respondents_who_engaged
   
3. Scale is 1-4 (not 1-5):
   - Very Traditional: 1
   - Traditional: 2
   - Progressive: 3
   - Very Progressive: 4

4. Methodology counts:
   - Total initiated: 496
   - After consent: 493
   - Completed national prosecutor section: 407 (82.1%)
   - Completed full survey: 365 (74.0%)

Author: Dvir Yogev, BERQ-J
Institution: Criminal Law & Justice Center, UC Berkeley Law
Date: October 2025
Version: FULLY CORRECTED - October 23, 2025

================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr, ttest_ind, mannwhitneyu, chi2_contingency
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pathlib import Path
import warnings
import re
import os
from collections import Counter
warnings.filterwarnings('ignore')

# ================================================================================
# NORMALIZATION UTILITIES FOR MATCHING
# ================================================================================

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

STATE_NAME_TO_ABBREV = {v: k for k, v in STATE_ABBREV_TO_NAME.items()}

NICKNAME_MAP = {
    'bob': 'robert', 'robert': 'bob',
    'bill': 'william', 'william': 'bill',
    'jim': 'james', 'james': 'jim',
    'joe': 'joseph', 'joseph': 'joe',
    'kim': 'kimberly', 'kimberly': 'kim',
    'mike': 'michael', 'michael': 'mike',
    'dan': 'daniel', 'daniel': 'dan',
    'dave': 'david', 'david': 'dave',
    'steve': 'steven', 'steven': 'steve',
    'tony': 'anthony', 'anthony': 'tony',
    'liz': 'elizabeth', 'beth': 'elizabeth', 'elizabeth': 'liz',
    'alex': 'alexander', 'alexander': 'alex',
    'sue': 'susan', 'susan': 'sue',
    'greg': 'gregory', 'gregory': 'greg'
}


def std_series(s):
    """Lowercase + collapse whitespace for consistent string comparisons."""
    return s.astype(str).str.lower().str.strip().str.replace(r"\s+", " ", regex=True)


def std_value(value):
    """Standardize a single value using std_series semantics."""
    return std_series(pd.Series([value])).iat[0] if isinstance(value, str) else ''


def normalize_state_name(value):
    """Return the full state name given either full name or abbreviation."""
    if not isinstance(value, str):
        return None
    v = value.strip()
    if not v:
        return None
    upper = v.upper()
    if upper in STATE_ABBREV_TO_NAME:
        return STATE_ABBREV_TO_NAME[upper]
    title = v.title()
    if title in STATE_NAME_TO_ABBREV:
        return title
    return title


def norm_first(value: str) -> str:
    """Normalize a first name for matching (nicknames + initials)."""
    if not isinstance(value, str) or not value.strip():
        return ''
    value = value.strip().lower()
    return NICKNAME_MAP.get(value, value)


def first_name_compatible(a: str, b: str) -> bool:
    """Check whether two first names are compatible (nickname/initial aware)."""
    a_norm = norm_first(a)
    b_norm = norm_first(b)
    if not a_norm or not b_norm:
        return True
    if a_norm == b_norm:
        return True
    if a_norm.startswith(b_norm) or b_norm.startswith(a_norm):
        return True
    return a_norm[0] == b_norm[0]


def clean_jurisdiction_for_matching(text: str) -> str:
    """Standardize jurisdiction strings (remove county, parentheticals, trim)."""
    if not isinstance(text, str):
        return ''
    cleaned = re.sub(r'\s*\([^)]*\)', '', text)
    cleaned = re.sub(r'County', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'Parish', '', cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace('Borough', '').replace('City', '')
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned.strip().lower()


def split_full_name(name: str):
    """Split a full name into (first, last) components."""
    if not isinstance(name, str) or not name.strip():
        return '', ''
    normalized = re.sub(r"[’]", "'", name.strip())
    parts = normalized.split()
    if len(parts) == 1:
        return parts[0], ''
    return ' '.join(parts[:-1]), parts[-1]

# Set up plotting parameters
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.unicode_minus'] = True
sns.set_style("whitegrid")

# ================================================================================
# CONFIGURATION
# ================================================================================

class Config:
    """Configuration settings for the analysis"""

    # File paths
    SURVEY_FILE = 'PP+survey+draft_October+16,+2025_13.32.csv'
    ELECTION_FILE = 'elections_with_reconciled_contested.csv'
    OUTPUT_DIR = 'outputs/corrected'
    VIZ_DIR = 'outputs/corrected/visualizations'

    # Analysis parameters
    MIN_RATINGS_THRESHOLD = 10  # Minimum ratings for reliable IDEOLOGY analysis
    CLOSE_MARGIN_THRESHOLD = 55  # % threshold for "close" elections
    PRIMARY_CLOSE_MARGIN_DIFFERENCE = 5  # fallback difference threshold if percent columns missing

    # Rating scale (CORRECTED to 1-4)
    RATING_MAP = {
        'Very Traditional': 1,
        'Traditional': 2,
        'Not Familiar': np.nan,
        'Progressive': 3,
        'Very Progressive': 4
    }

    # DA names mapping for 50 nationally notable prosecutors
    DA_NAMES = {
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
        'notable_46': ('Glenn Funk', 'Davidson County (Nashville), TN'),
        'notable_47': ('Karl Racine', 'Washington, DC'),
        'notable_48': ('Brian Schwalb', 'Washington, DC'),
        'notable_49': ('Marilyn Mosby', 'Baltimore City, MD'),
        'notable_50': ('Ivan Bates', 'Baltimore City, MD')
    }
    
    # Recalled prosecutors
    RECALLED_PROSECUTORS = ['Chesa Boudin', 'Pamela Price']

    # Known progressive/traditional categorizations
    PROGRESSIVE_NAMES = ['Larry Krasner', 'Chesa Boudin', 'George Gascon', 'Pamela Price',
                         'Mary Moriarty', 'Kim Foxx', 'Kim Gardner', 'Jose Garza',
                         'Monique Worrell', 'Andrew Warren', 'Mike Schmidt']
    TRADITIONAL_NAMES = ['Rachel Mitchell', 'Amy Weirich', 'Brooke Jenkins', 'Nathan Hochman',
                        'Summer Stephan', 'Katherine Fernandez-Rundle']

    # Region groupings for extended visibility analyses
    REGION_MAP = {
        'West Coast (CA, OR, WA)': {'California', 'Oregon', 'Washington'},
        'Northeast (MA, NY, PA, MD)': {'Massachusetts', 'New York', 'Pennsylvania', 'Maryland'},
        'South (FL, GA, TX, TN)': {'Florida', 'Georgia', 'Texas', 'Tennessee'},
        'Midwest (IL, IN, MI, OH)': {'Illinois', 'Indiana', 'Michigan', 'Ohio'},
        'Mountain West (AZ, CO, UT)': {'Arizona', 'Colorado', 'Utah'},
        'Deep South (AL, MS, LA)': {'Alabama', 'Mississippi', 'Louisiana'}
    }

    BIDEN_STATES_2020 = {
        'Arizona', 'California', 'Colorado', 'Connecticut', 'Delaware', 'District of Columbia',
        'Georgia', 'Hawaii', 'Illinois', 'Maine', 'Maryland', 'Massachusetts', 'Michigan',
        'Minnesota', 'Nevada', 'New Hampshire', 'New Jersey', 'New Mexico', 'New York',
        'Oregon', 'Pennsylvania', 'Rhode Island', 'Vermont', 'Virginia', 'Washington',
        'Wisconsin'
    }

    TRUMP_STATES_2020 = {
        'Alabama', 'Alaska', 'Arkansas', 'Florida', 'Georgia', 'Idaho', 'Indiana', 'Iowa',
        'Kansas', 'Kentucky', 'Louisiana', 'Mississippi', 'Missouri', 'Montana', 'Nebraska',
        'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma', 'South Carolina', 'South Dakota',
        'Tennessee', 'Texas', 'Utah', 'West Virginia', 'Wyoming'
    }

    # County-level population lookup for notable prosecutors (2020 estimates)
    NOTABLE_COUNTY_POPULATION = {
        'Larry Krasner': ('Philadelphia County, PA', 1603797),
        'Alvin Bragg': ('New York County, NY', 1689153),
        'Mary Moriarty': ('Hennepin County, MN', 1265843),
        'Brooke Jenkins': ('San Francisco County, CA', 873965),
        'Chesa Boudin': ('San Francisco County, CA', 873965),
        'Pamela Price': ('Alameda County, CA', 1671329),
        "Nancy O'Malley": ('Alameda County, CA', 1671329),
        'George Gascon': ('Los Angeles County, CA', 10039107),
        'Nathan Hochman': ('Los Angeles County, CA', 10039107),
        'Kim Foxx': ('Cook County, IL', 5246459),
        "Eileen O'Neill Burke": ('Cook County, IL', 5246459),
        'Kim Gardner': ('St. Louis City, MO', 301578),
        'Monique Worrell': ('Orange County, FL', 1445003),
        'Andrew Warren': ('Hillsborough County, FL', 1505064),
        'Michael Dougherty': ('Boulder County, CO', 330758),
        'Sim Gill': ('Salt Lake County, UT', 1181849),
        'Eric Gonzalez': ('Kings County, NY', 2559903),
        'Eli Savit': ('Washtenaw County, MI', 372258),
        'Kym Worthy': ('Wayne County, MI', 1749343),
        'Summer Stephan': ('San Diego County, CA', 3338330),
        'Katherine Fernandez-Rundle': ('Miami-Dade County, FL', 2716940),
        'Melissa Nelson': ('Duval County, FL', 995567),
        'Mark Dupree': ('Wyandotte County, KS', 165429),
        'Jose Garza': ('Travis County, TX', 1295177),
        'John Creuzot': ('Dallas County, TX', 2635516),
        'Kim Ogg': ('Harris County, TX', 4713325),
        'Sean Teare': ('Harris County, TX', 4713325),
        'Joe Gonzales': ('Bexar County, TX', 2003554),
        'Rachael Rollins': ('Suffolk County, MA', 797936),
        'Kevin Hayden': ('Suffolk County, MA', 797936),
        'Ryan Mears': ('Marion County, IN', 977203),
        'Dan Satterberg': ('King County, WA', 2252782),
        'Leesa Manion': ('King County, WA', 2252782),
        'Jeff Rosen': ('Santa Clara County, CA', 1937570),
        'Rachel Mitchell': ('Maricopa County, AZ', 4485414),
        'Laura Conover': ('Pima County, AZ', 1047279),
        'Mike Schmidt': ('Multnomah County, OR', 815428),
        'Nathan Vasquez': ('Multnomah County, OR', 815428),
        'Amy Weirich': ('Shelby County, TN', 929744),
        'Steve Mulroy': ('Shelby County, TN', 929744),
        'Fani Willis': ('Fulton County, GA', 1063937),
        'Sherry Boston': ('DeKalb County, GA', 764382),
        'Satana Deberry': ('Durham County, NC', 324833),
        'Jason Williams': ('Orleans Parish, LA', 383997),
        "Michael O'Malley": ('Cuyahoga County, OH', 1249352),
        'Glenn Funk': ('Davidson County, TN', 715884),
        'Karl Racine': ('District of Columbia', 689545),
        'Brian Schwalb': ('District of Columbia', 689545),
        'Marilyn Mosby': ('Baltimore City, MD', 585708),
        'Ivan Bates': ('Baltimore City, MD', 585708)
    }

    POPULATION_CATEGORY_LABELS = {
        'large': 'Large urban counties (>1M population)',
        'mid': 'Mid-size counties (250K-1M)',
        'small': 'Smaller jurisdictions (<250K)'
    }


# ================================================================================
# UTILITY FUNCTIONS
# ================================================================================

def print_section(title, char='='):
    """Print a formatted section header"""
    print("\n" + char * 80)
    print(title)
    print(char * 80)
    print()


def ensure_dir(directory):
    """Create directory if it doesn't exist"""
    Path(directory).mkdir(parents=True, exist_ok=True)


# ================================================================================
# DATA LOADER CLASS
# ================================================================================

class DataLoader:
    """Load and prepare survey and election data with CORRECTED familiarity calculations"""
    
    def __init__(self, survey_file, election_file):
        self.survey_file = survey_file
        self.election_file = election_file
        
    def load(self):
        """Load all data"""
        print_section("LOADING DATA")
        
        # Load survey data (Qualtrics format)
        try:
            self.df_survey = pd.read_csv(self.survey_file, skiprows=[1])
            self.questions_df = pd.read_csv(self.survey_file, nrows=1)
            print(f"✓ Loaded {len(self.df_survey)} survey responses")
        except:
            self.df_survey = pd.read_csv(self.survey_file)
            self.questions_df = None
            print(f"✓ Loaded {len(self.df_survey)} survey responses (standard format)")
        
        # Filter for consent
        if 'con' in self.df_survey.columns:
            self.df_survey = self.df_survey[self.df_survey['con'] == 'Agree'].copy()
            print(f"✓ After consent filter: {len(self.df_survey)} respondents")
        
        self.total_respondents = len(self.df_survey)
        
        # CRITICAL: Calculate who completed the national prosecutor section
        # This is determined by who answered the 'more' question (comes after all notable prosecutors)
        if 'more' in self.df_survey.columns:
            self.completed_national_section = self.df_survey['more'].notna().sum()
            print(f"✓ Completed national prosecutor section: {self.completed_national_section} ({self.completed_national_section/self.total_respondents*100:.1f}%)")
        else:
            # Fallback: anyone who responded to at least one notable prosecutor
            self.completed_national_section = self.total_respondents
            print(f"⚠ Warning: 'more' column not found, using all respondents for national section")
        
        # CRITICAL: Get state-level respondent counts
        self.state_respondent_counts = {}
        if 'RespondentState' in self.df_survey.columns:
            state_counts = self.df_survey['RespondentState'].value_counts()
            self.state_respondent_counts = state_counts.to_dict()
            print(f"✓ Identified {len(self.state_respondent_counts)} states with respondents")
            print(f"  Top 5 states: {dict(list(state_counts.head(5).items()))}")
        
        # Load election data
        self.df_election = pd.read_csv(self.election_file)
        print(f"✓ Loaded {len(self.df_election)} election records")
        
        # Identify columns
        self.notable_cols = [col for col in self.df_survey.columns if col.startswith('notable_')]
        print(f"✓ Identified {len(self.notable_cols)} notable prosecutor columns")
        
        # Identify state DA columns
        self.state_da_cols = []
        for col in self.df_survey.columns:
            if '_' in col and not col.startswith('Q') and not col.startswith('notable'):
                # Must be State_Number format
                parts = col.split('_')
                if len(parts) == 2 and parts[0] not in ['gender', 'race', 'Q55']:
                    # Check if it contains rating responses
                    unique_vals = self.df_survey[col].dropna().unique()
                    if len(unique_vals) > 0 and any('Progressive' in str(v) or 'Traditional' in str(v) 
                                                     for v in unique_vals):
                        self.state_da_cols.append(col)
                        # Extract state name
                        state_name = parts[0]
                        if state_name not in self.state_respondent_counts:
                            # State has prosecutors but no respondents - shouldn't happen but be safe
                            self.state_respondent_counts[state_name] = 0
        
        print(f"✓ Identified {len(self.state_da_cols)} state prosecutor columns")

        return self

    # ------------------------------------------------------------------
    # Metadata extraction helpers (names + jurisdictions)
    # ------------------------------------------------------------------
    def _get_question_label(self, col: str) -> str:
        if self.questions_df is not None and col in self.questions_df.columns:
            return str(self.questions_df[col].iloc[0])
        return ''

    def extract_notable_metadata(self, col: str) -> dict:
        label = self._get_question_label(col)
        tail = label.split(' - ')[-1].strip() if ' - ' in label else label.strip()
        if not tail and col in Config.DA_NAMES:
            name, jurisdiction = Config.DA_NAMES[col]
        else:
            if ',' in tail:
                name, jurisdiction = tail.split(',', 1)
                name = name.strip()
                jurisdiction = jurisdiction.strip()
            else:
                name = tail.strip()
                jurisdiction = ''
        first, last = split_full_name(name)
        state_abbrev = None
        state_full = None
        if jurisdiction:
            parts = [p.strip() for p in jurisdiction.split(',') if p.strip()]
            if parts:
                poss_state = parts[-1]
                state_full = normalize_state_name(poss_state)
                if state_full and state_full in STATE_NAME_TO_ABBREV:
                    state_abbrev = STATE_NAME_TO_ABBREV[state_full]
        meta = {
            'column': col,
            'scope': 'national',
            'is_notable': True,
            'name': name,
            'fname': first,
            'lname': last,
            'jurisdiction': jurisdiction,
            'jurisdiction_clean': clean_jurisdiction_for_matching(jurisdiction),
            'state_name': state_full,
            'state_abbrev': state_abbrev,
            'state_source': None
        }
        return meta

    def extract_state_metadata(self, col: str) -> dict:
        state_token = col.split('_')[0]
        state_full = normalize_state_name(state_token) or state_token
        state_abbrev = STATE_NAME_TO_ABBREV.get(state_full)
        label = self._get_question_label(col)
        tail = label.split(' - ')[-1].strip() if ' - ' in label else label.strip()
        parts = [p.strip() for p in tail.split('\t') if p.strip()]
        lname = parts[0] if len(parts) >= 1 else ''
        fname = parts[1] if len(parts) >= 2 else ''
        jurisdiction_core = parts[2] if len(parts) >= 3 else ''
        if fname and lname:
            name = f"{fname} {lname}".strip()
        else:
            name = tail if tail else col
        first, last = split_full_name(name)
        jurisdiction_display = ''
        if jurisdiction_core:
            suffix_state = state_abbrev or state_full
            jurisdiction_display = f"{jurisdiction_core} County, {suffix_state}" if suffix_state else jurisdiction_core
        meta = {
            'column': col,
            'scope': 'state',
            'is_notable': False,
            'name': name,
            'fname': first if first else fname,
            'lname': last if last else lname,
            'jurisdiction': jurisdiction_display,
            'jurisdiction_clean': clean_jurisdiction_for_matching(jurisdiction_core or jurisdiction_display or state_full),
            'state_name': state_full,
            'state_abbrev': state_abbrev,
            'state_source': state_token
        }
        return meta


# ================================================================================
# ELECTION MATCHER
# ================================================================================

class ElectionMatcher:
    """Robustly match survey prosecutors to election records."""

    def __init__(self, elections_df: pd.DataFrame):
        self.df = elections_df.copy()
        self.df['cand_fname'] = self.df['cand_fname'].fillna('').astype(str)
        self.df['cand_lname'] = self.df['cand_lname'].fillna('').astype(str)
        self.df['state'] = self.df['state'].fillna('').astype(str)
        self.df['std_fname'] = std_series(self.df['cand_fname'])
        self.df['std_lname'] = std_series(self.df['cand_lname'])
        self.df['std_state'] = std_series(self.df['state'])
        self.df['election_year'] = pd.to_numeric(self.df['election_year'], errors='coerce')
        self.df['jurisdiction_tokens'] = self.df.apply(self._compute_jurisdiction_tokens, axis=1)

    def _compute_jurisdiction_tokens(self, row) -> frozenset:
        tokens = set()
        for col in ('district', 'counties_total'):
            val = row.get(col)
            if isinstance(val, str) and val.strip():
                parts = [p.strip() for p in val.split(',') if p.strip()]
                for part in parts:
                    cleaned = clean_jurisdiction_for_matching(part)
                    if cleaned:
                        tokens.add(cleaned)
        if not tokens:
            tokens.add('')
        return frozenset(tokens)

    def find_matches(self, prosecutor_row: pd.Series):
        """Return election rows (all cycles) for the best-matching candidate."""

        lname = std_value(prosecutor_row.get('lname'))
        if not lname:
            return pd.DataFrame(), 'no_match', False

        fname = std_value(prosecutor_row.get('fname'))
        juris = clean_jurisdiction_for_matching(prosecutor_row.get('jurisdiction_clean') or prosecutor_row.get('jurisdiction'))

        state_options = []
        for state_candidate in [prosecutor_row.get('state_name'), prosecutor_row.get('state_source'), prosecutor_row.get('state_abbrev')]:
            if state_candidate is None or pd.isna(state_candidate):
                continue
            full_name = normalize_state_name(state_candidate) if state_candidate and len(str(state_candidate)) <= 3 else state_candidate
            normalized = normalize_state_name(full_name)
            std_state = std_value(normalized)
            if std_state and std_state not in state_options:
                state_options.append(std_state)
        state_options.append(None)  # allow fallback without state filter

        best_matches = pd.DataFrame()
        best_method = 'no_match'
        fname_matched = False
        selected_state = None

        for state_norm in state_options:
            candidates = self.df
            method_prefix = 'any_state'
            if state_norm:
                candidates = candidates[candidates['std_state'] == state_norm]
                method_prefix = 'state'
            candidates = candidates[candidates['std_lname'] == lname]
            if candidates.empty:
                continue

            method_candidate = f"{method_prefix}_lname"
            filtered = candidates
            if juris:
                juris_mask = candidates['jurisdiction_tokens'].apply(lambda tokens: juris in tokens)
                if juris_mask.any():
                    filtered = candidates[juris_mask]
                    method_candidate = f"{method_candidate}_juris"

            if filtered.empty:
                filtered = candidates

            if fname:
                compat_mask = filtered['std_fname'].apply(lambda x: first_name_compatible(fname, x))
                if compat_mask.any():
                    filtered = filtered[compat_mask]
                    method_candidate = f"{method_candidate}_fname"
                    fname_matched = True
                else:
                    fname_matched = False
            else:
                fname_matched = True

            if not filtered.empty:
                best_matches = filtered.copy()
                best_method = method_candidate
                selected_state = state_norm
                break

        if best_matches.empty:
            return pd.DataFrame(), 'no_match', False

        # Choose candidate with the most recent election
        if 'candidate_unique_identifier' in best_matches.columns and best_matches['candidate_unique_identifier'].notna().any():
            grouped = best_matches.groupby('candidate_unique_identifier')['election_year'].max()
            best_id = grouped.idxmax()
            matches = self.df[self.df['candidate_unique_identifier'] == best_id].copy()
        else:
            latest_year = best_matches['election_year'].max()
            mask = (self.df['std_lname'] == lname) & (self.df['election_year'] == latest_year)
            if selected_state:
                mask &= self.df['std_state'] == selected_state
            matches = self.df[mask].copy()
            if matches.empty:
                matches = best_matches.copy()

        matches = matches.sort_values('election_year', ascending=False)
        matches['match_method'] = best_method
        matches['first_name_compatible'] = 'Y' if fname_matched else 'N'
        return matches, best_method, fname_matched

# ================================================================================
# PROSECUTOR ANALYZER CLASS
# ================================================================================

class ProsecutorAnalyzer:
    """Main analyzer class with CORRECTED familiarity calculations"""
    
    def __init__(self, data_loader):
        self.loader = data_loader
        self.df_survey = data_loader.df_survey
        self.df_election = data_loader.df_election
        self.questions_df = data_loader.questions_df
        self.notable_cols = data_loader.notable_cols
        self.state_da_cols = data_loader.state_da_cols
        self.total_respondents = data_loader.total_respondents
        self.completed_national_section = data_loader.completed_national_section
        self.state_respondent_counts = data_loader.state_respondent_counts
        
        # Results storage
        self.results = {}

    # ------------------------------------------------------------------
    # Helper utilities for extended analyses
    # ------------------------------------------------------------------

    def _lookup_county_population(self, row):
        if not row.get('is_notable'):
            return np.nan
        name = row.get('name')
        info = Config.NOTABLE_COUNTY_POPULATION.get(name)
        if info:
            return info[1]
        return np.nan

    def _population_category_label(self, population):
        if pd.isna(population):
            return np.nan
        if population >= 1_000_000:
            return Config.POPULATION_CATEGORY_LABELS['large']
        if population >= 250_000:
            return Config.POPULATION_CATEGORY_LABELS['mid']
        return Config.POPULATION_CATEGORY_LABELS['small']

    def _classify_position(self, value):
        """Map raw position responses into consistent professional groups."""

        if pd.isna(value):
            return 'Other professionals'
        text = str(value).strip().lower()
        if not text:
            return 'Other professionals'
        if 'prosecutor' in text or 'district attorney' in text:
            return 'Prosecutor'
        if 'defense' in text:
            return 'Defense Attorney'
        if 'academic' in text or 'professor' in text:
            return 'Academic'
        return 'Other professionals'

    def _assign_region(self, state_value):
        if not isinstance(state_value, str) or not state_value:
            return 'Other regions'
        normalized = normalize_state_name(state_value)
        for label, states in Config.REGION_MAP.items():
            if normalized in states:
                return label
        return 'Other regions'

    def analyze_prosecutor_column(self, col, metadata: dict):
        """Analyze a single prosecutor column with CORRECTED familiarity calculation."""

        col_name = metadata.get('name', col)
        scope = metadata.get('scope', 'national')
        state_source = metadata.get('state_source')
        responses = self.df_survey[col].dropna()

        # Determine denominator based on scope
        if scope == 'state' and state_source and 'RespondentState' in self.df_survey.columns:
            state_mask = self.df_survey['RespondentState'] == state_source
            state_respondents = self.df_survey[state_mask]
            state_cols_for_this_state = [c for c in self.state_da_cols if c.startswith(state_source + '_')]

            engaged_count = 0
            for idx in state_respondents.index:
                if any(pd.notna(self.df_survey.loc[idx, c]) for c in state_cols_for_this_state):
                    engaged_count += 1

            total_responses = engaged_count if engaged_count > 0 else self.state_respondent_counts.get(state_source, 0)
        else:
            total_responses = self.completed_national_section

        numeric_ratings = responses.map(Config.RATING_MAP)
        substantive_ratings = numeric_ratings.dropna()

        if len(substantive_ratings) == 0:
            return None

        familiarity_rate = (len(substantive_ratings) / total_responses) * 100 if total_responses > 0 else 0

        column_series = self.df_survey[col]
        very_traditional_count = int((column_series == 'Very Traditional').sum())
        traditional_count = int((column_series == 'Traditional').sum())
        progressive_count = int((column_series == 'Progressive').sum())
        very_progressive_count = int((column_series == 'Very Progressive').sum())
        not_familiar_count = int((column_series == 'Not Familiar').sum())
        blank_count = int(column_series.isna().sum())

        # Position-specific means
        academic_mean = defense_mean = prosecutor_mean = other_mean = np.nan
        if 'position' in self.df_survey.columns:
            position_df = pd.DataFrame({
                'position': self.df_survey['position'],
                'rating': column_series
            })
            position_df['score'] = position_df['rating'].map(Config.RATING_MAP)
            position_df = position_df.dropna(subset=['score'])
            if not position_df.empty:
                position_df['position_group'] = position_df['position'].apply(self._classify_position)
                group_means = position_df.groupby('position_group')['score'].mean()
                academic_mean = group_means.get('Academic', np.nan)
                defense_mean = group_means.get('Defense Attorney', np.nan)
                prosecutor_mean = group_means.get('Prosecutor', np.nan)
                other_mean = group_means.get('Other professionals', np.nan)

        result = {
            'column': col,
            'name': col_name,
            'scope': scope,
            'state_display': metadata.get('state_name') or metadata.get('state_source') or 'National',
            'state': metadata.get('state_name') or metadata.get('state_source') or 'National',
            'state_name': metadata.get('state_name'),
            'state_abbrev': metadata.get('state_abbrev'),
            'state_source': state_source,
            'total_responses': total_responses,
            'substantive_ratings': len(substantive_ratings),
            'familiarity_rate': familiarity_rate,
            'mean_score': substantive_ratings.mean(),
            'median_score': substantive_ratings.median(),
            'std_score': substantive_ratings.std(),
            'fname': metadata.get('fname'),
            'lname': metadata.get('lname'),
            'jurisdiction': metadata.get('jurisdiction'),
            'jurisdiction_clean': metadata.get('jurisdiction_clean'),
            'is_notable': metadata.get('is_notable', False),
            'very_traditional_count': very_traditional_count,
            'traditional_count': traditional_count,
            'progressive_count': progressive_count,
            'very_progressive_count': very_progressive_count,
            'not_familiar_count': not_familiar_count,
            'blank_count': blank_count,
            'academic_mean': academic_mean,
            'defense_mean': defense_mean,
            'prosecutor_mean': prosecutor_mean,
            'other_position_mean': other_mean
        }

        if len(substantive_ratings) > 0:
            result['very_traditional_pct'] = (very_traditional_count / len(substantive_ratings)) * 100
            result['traditional_pct'] = (traditional_count / len(substantive_ratings)) * 100
            result['progressive_pct'] = (progressive_count / len(substantive_ratings)) * 100
            result['very_progressive_pct'] = (very_progressive_count / len(substantive_ratings)) * 100
        else:
            result['very_traditional_pct'] = result['traditional_pct'] = np.nan
            result['progressive_pct'] = result['very_progressive_pct'] = np.nan

        return result
    
    def build_prosecutor_dataset(self):
        """Build comprehensive prosecutor dataset with CORRECTED familiarity"""
        print_section("BUILDING PROSECUTOR DATASET (CORRECTED FAMILIARITY)")
        
        all_prosecutors = []
        
        # Process notable prosecutors (all respondents as denominator)
        print("Processing notable prosecutors...")
        for col in self.notable_cols:
            meta = self.loader.extract_notable_metadata(col)
            if not meta.get('name'):
                continue
            result = self.analyze_prosecutor_column(col, meta)
            if result:
                result['location'] = meta.get('jurisdiction') or meta.get('state_name') or 'National'
                all_prosecutors.append(result)

        print(f"✓ Processed {len([p for p in all_prosecutors if p.get('is_notable')])} notable prosecutors")
        
        # Process state prosecutors (state-specific denominators)
        print("\nProcessing state prosecutors with CORRECTED denominators...")
        state_counts = {}
        
        for col in self.state_da_cols:
            meta = self.loader.extract_state_metadata(col)
            if not meta.get('name'):
                continue
            result = self.analyze_prosecutor_column(col, meta)
            if result:
                result['location'] = meta.get('jurisdiction') or meta.get('state_name') or meta.get('state_source')
                all_prosecutors.append(result)

                state_key = meta.get('state_source') or meta.get('state_name') or 'Unknown'
                state_counts[state_key] = state_counts.get(state_key, 0) + 1

        print(f"✓ Processed {len([p for p in all_prosecutors if not p.get('is_notable')])} state prosecutors")
        print(f"  Across {len(state_counts)} states")
        print(f"  Top 5 states by prosecutors: {dict(sorted(state_counts.items(), key=lambda x: x[1], reverse=True)[:5])}")

        self.df_prosecutors = pd.DataFrame(all_prosecutors)
        
        print(f"\n✓ Created dataset with {len(self.df_prosecutors)} prosecutors")
        print(f"  - {len(self.df_prosecutors[self.df_prosecutors['is_notable']])} notable")
        print(f"  - {len(self.df_prosecutors[~self.df_prosecutors['is_notable']])} state")
        
        # Show familiarity comparison
        notable_fam = self.df_prosecutors[self.df_prosecutors['is_notable']]['familiarity_rate'].mean()
        state_fam = self.df_prosecutors[~self.df_prosecutors['is_notable']]['familiarity_rate'].mean()
        
        print(f"\n📊 CORRECTED FAMILIARITY RATES:")
        print(f"  Notable prosecutors: {notable_fam:.2f}% mean familiarity")
        print(f"  State prosecutors: {state_fam:.2f}% mean familiarity")
        print(f"  Difference: {notable_fam - state_fam:+.2f} percentage points")
        
        # Filter for meaningful IDEOLOGY ratings (≥10 for reliable mean scores)
        self.df_prosecutors_filtered_ideology = self.df_prosecutors[
            self.df_prosecutors['substantive_ratings'] >= Config.MIN_RATINGS_THRESHOLD
        ].copy()
        print(f"\n✓ Filtered to {len(self.df_prosecutors_filtered_ideology)} prosecutors with ≥{Config.MIN_RATINGS_THRESHOLD} ratings")
        print(f"  (Used for ideology-based analyses)")
        
        self.results['all_prosecutors'] = self.df_prosecutors
        self.results['prosecutors_filtered_ideology'] = self.df_prosecutors_filtered_ideology

        # Prepare supplemental aggregates used for legacy visualization suite
        self.build_state_summary()
        self.build_transition_summary()

        # Store composition breakdowns for master reporting
        if 'position' in self.df_survey.columns:
            position_series = self.df_survey['position'].apply(self._classify_position)
            position_map = {
                'Prosecutor': 'Prosecutors',
                'Defense Attorney': 'Defense Attorneys',
                'Academic': 'Academics'
            }
            mapped_positions = position_series.map(lambda x: position_map.get(x, 'Other Criminal Justice Professionals'))
            position_counts = mapped_positions.value_counts().to_dict()
        else:
            position_counts = {}

        self.results['position_breakdown'] = position_counts
        self.results['state_prosecutor_counts'] = state_counts

        return self

    def build_state_summary(self):
        """Aggregate state-level prosecutor ratings for visualization reuse."""

        state_results = {}
        for col in self.state_da_cols:
            meta = self.loader.extract_state_metadata(col)
            state_key = meta.get('state_source') or meta.get('state_name')
            if not state_key:
                continue
            ratings = self.df_survey[col].map(Config.RATING_MAP).dropna()
            if ratings.empty:
                continue
            bucket = state_results.setdefault(state_key, {
                'state': state_key,
                'ratings': [],
                'prosecutor_count': 0
            })
            bucket['ratings'].extend(ratings.tolist())
            bucket['prosecutor_count'] += 1

        summary_rows = []
        for state_key, data in state_results.items():
            arr = np.array(data['ratings'])
            if len(arr) < Config.MIN_RATINGS_THRESHOLD:
                continue
            summary_rows.append({
                'state': state_key,
                'n_prosecutors': data['prosecutor_count'],
                'total_ratings': len(arr),
                'mean_score': arr.mean(),
                'median_score': np.median(arr),
                'std_score': arr.std(ddof=0),
                'progressive_pct': (arr >= 3).sum() / len(arr) * 100
            })

        self.state_df = pd.DataFrame(summary_rows).sort_values('mean_score', ascending=False)
        self.results['state_summary'] = self.state_df

    def build_transition_summary(self):
        """Recreate jurisdiction transition comparisons used in legacy plots."""

        transitions = [
            ('notable_5', 'notable_4', 'San Francisco, CA'),
            ('notable_7', 'notable_6', 'Alameda County, CA'),
            ('notable_8', 'notable_9', 'Los Angeles, CA'),
            ('notable_10', 'notable_11', 'Cook County, IL'),
            ('notable_32', 'notable_33', 'King County, WA'),
            ('notable_29', 'notable_30', 'Suffolk County, MA'),
            ('notable_37', 'notable_38', 'Multnomah County, OR'),
            ('notable_39', 'notable_40', 'Shelby County, TN'),
            ('notable_26', 'notable_27', 'Harris County, TX'),
            ('notable_47', 'notable_48', 'Washington, DC'),
            ('notable_49', 'notable_50', 'Baltimore City, MD')
        ]

        rows = []
        for predecessor, successor, jurisdiction in transitions:
            pred_row = self.df_prosecutors[self.df_prosecutors['column'] == predecessor]
            succ_row = self.df_prosecutors[self.df_prosecutors['column'] == successor]

            if pred_row.empty or succ_row.empty:
                continue

            pred_score = pred_row['mean_score'].iloc[0]
            succ_score = succ_row['mean_score'].iloc[0]

            if pd.isna(pred_score) or pd.isna(succ_score):
                continue

            rows.append({
                'Jurisdiction': jurisdiction,
                'Predecessor': pred_row['name'].iloc[0],
                'Successor': succ_row['name'].iloc[0],
                'Pred_Score': pred_score,
                'Succ_Score': succ_score,
                'Change': succ_score - pred_score,
                'Direction': '→ Progressive' if succ_score - pred_score > 0 else '→ Traditional'
            })

        self.transitions_df = pd.DataFrame(rows).sort_values('Change')
        self.results['transitions'] = self.transitions_df
    
    def match_with_elections(self):
        """Match prosecutors with election data"""
        print_section("MATCHING WITH ELECTION DATA")

        matcher = ElectionMatcher(self.df_election)

        print("Creating FULL matched sample (all prosecutors, any # of ratings)...")
        matched_data_all = []
        unmatched = []

        for _, row in self.df_prosecutors.iterrows():
            matches, method, fname_matched = matcher.find_matches(row)

            if matches.empty:
                unmatched.append(row['name'])
                continue

            contested_primary = matches['primary_contested_reconciled'].astype(str).str.lower() == 'contested'
            contested_general = matches['general_contested_reconciled'].astype(str).str.lower() == 'contested'
            ever_incumbent = matches['incum_chall'].astype(str).str.upper() == 'I'
            ever_challenger = matches['incum_chall'].astype(str).str.upper() == 'C'

            years = sorted(matches['election_year'].dropna().unique().tolist())
            primary_margins = pd.to_numeric(matches.loc[matches['winner_primary'] == 'W', 'vote_percent_primary'], errors='coerce').dropna()
            general_margins = pd.to_numeric(matches.loc[matches['winner_general'] == 'W', 'vote_percent_general'], errors='coerce').dropna()

            if 'election_unique_identifier' in matches.columns:
                num_elections = matches['election_unique_identifier'].nunique()
            else:
                num_elections = int(matches['election_year'].notna().sum())

            election_stats = {
                'num_elections': int(num_elections),
                'years': years,
                'ever_contested_primary': contested_primary.any(),
                'ever_contested_general': contested_general.any(),
                'ever_contested_any': contested_primary.any() or contested_general.any(),
                'ever_ran_as_incumbent': ever_incumbent.any(),
                'ever_ran_as_challenger': ever_challenger.any(),
                'closest_primary_margin': primary_margins.min() if len(primary_margins) else np.nan,
                'closest_general_margin': general_margins.min() if len(general_margins) else np.nan,
                'had_close_primary': (primary_margins < Config.CLOSE_MARGIN_THRESHOLD).any() if len(primary_margins) else False,
                'had_close_general': (general_margins < Config.CLOSE_MARGIN_THRESHOLD).any() if len(general_margins) else False,
                'latest_election_year': max(years) if years else np.nan,
                'match_method': method,
                'first_name_compatible': 'Y' if fname_matched else 'N'
            }

            if 'candidate_unique_identifier' in matches.columns:
                valid_ids = matches['candidate_unique_identifier'].dropna()
                if not valid_ids.empty:
                    election_stats['candidate_unique_identifier'] = valid_ids.iloc[0]

            result = {**row.to_dict(), **election_stats}
            matched_data_all.append(result)

        self.df_matched_all = pd.DataFrame(matched_data_all)

        if unmatched:
            print(f"⚠ Unmatched prosecutors: {len(unmatched)}")

        # Attach population metadata for notable prosecutors
        if not self.df_matched_all.empty:
            self.df_matched_all['county_population'] = self.df_matched_all.apply(
                self._lookup_county_population, axis=1
            )
            self.df_matched_all['population_category'] = self.df_matched_all['county_population'].apply(
                self._population_category_label
            )

        print(f"✓ FULL matched sample: {len(self.df_matched_all)} prosecutors")
        print(f"  - {len(self.df_matched_all[self.df_matched_all['is_notable']])} notable prosecutors")
        print(f"  - {len(self.df_matched_all[~self.df_matched_all['is_notable']])} state prosecutors")
        print(f"  - Use for: Familiarity analyses, incumbency comparisons")
        
        # FILTERED SAMPLE (≥10 ratings) for ideology analyses
        print(f"\nCreating FILTERED matched sample (≥{Config.MIN_RATINGS_THRESHOLD} ratings only)...")
        self.df_matched = self.df_matched_all[
            self.df_matched_all['substantive_ratings'] >= Config.MIN_RATINGS_THRESHOLD
        ].copy()
        
        print(f"✓ FILTERED matched sample: {len(self.df_matched)} prosecutors")
        print(f"  - {len(self.df_matched[self.df_matched['is_notable']])} notable prosecutors")
        print(f"  - {len(self.df_matched[~self.df_matched['is_notable']])} state prosecutors")
        print(f"  - Use for: Ideology analyses, multivariate models")
        
        # Add ideology categories (only for filtered sample with reliable scores)
        self.df_matched['ideology_category'] = pd.cut(
            self.df_matched['mean_score'],
            bins=[0, 2.0, 3.0, 5.0],
            labels=['Traditional', 'Moderate', 'Progressive']
        )

        if not self.df_matched.empty:
            if 'county_population' not in self.df_matched.columns:
                self.df_matched['county_population'] = self.df_matched_all.set_index('name').reindex(
                    self.df_matched['name']
                )['county_population'].values
            if 'population_category' not in self.df_matched.columns:
                self.df_matched['population_category'] = self.df_matched_all.set_index('name').reindex(
                    self.df_matched['name']
                )['population_category'].values

        # Add manual categorizations for RQ analyses
        self.df_matched['is_progressive'] = self.df_matched['name'].isin(Config.PROGRESSIVE_NAMES)
        self.df_matched['is_traditional'] = self.df_matched['name'].isin(Config.TRADITIONAL_NAMES)

        self.df_matched_all['is_progressive'] = self.df_matched_all['name'].isin(Config.PROGRESSIVE_NAMES)
        self.df_matched_all['is_traditional'] = self.df_matched_all['name'].isin(Config.TRADITIONAL_NAMES)

        # Add recalled indicator
        self.df_matched['recalled'] = self.df_matched['name'].isin(Config.RECALLED_PROSECUTORS)
        self.df_matched_all['recalled'] = self.df_matched_all['name'].isin(Config.RECALLED_PROSECUTORS)

        self.results['matched_all'] = self.df_matched_all
        self.results['matched_filtered'] = self.df_matched

        # Summaries for integration rates
        total_notable = int(self.df_prosecutors['is_notable'].sum())
        total_state = int((~self.df_prosecutors['is_notable']).sum())
        matched_notable = int(self.df_matched_all[self.df_matched_all['is_notable']].shape[0])
        matched_state = int(self.df_matched_all[~self.df_matched_all['is_notable']].shape[0])
        unmatched_set = set(unmatched)
        notable_names = set(self.df_prosecutors[self.df_prosecutors['is_notable']]['name'])
        state_names = set(self.df_prosecutors[~self.df_prosecutors['is_notable']]['name'])
        unmatched_notables = sorted(name for name in unmatched_set if name in notable_names)
        unmatched_states = sorted(name for name in unmatched_set if name in state_names)

        self.results['matching_summary'] = {
            'total_notable': total_notable,
            'matched_notable': matched_notable,
            'match_rate_notable': matched_notable / total_notable if total_notable else np.nan,
            'total_state': total_state,
            'matched_state': matched_state,
            'match_rate_state': matched_state / total_state if total_state else np.nan,
            'unmatched_notables': unmatched_notables,
            'unmatched_states': unmatched_states
        }

        return self
    
    def analyze_familiarity_patterns(self):
        """Analyze familiarity patterns with CORRECTED calculations"""
        print_section("ANALYZING FAMILIARITY PATTERNS (CORRECTED)")
        
        # Notable vs state comparison
        notable = self.df_prosecutors[self.df_prosecutors['is_notable']]
        state = self.df_prosecutors[~self.df_prosecutors['is_notable']]
        
        print("CORRECTED FAMILIARITY RATES:")
        print(f"  Notable prosecutors (n={len(notable)}):")
        print(f"    Mean familiarity: {notable['familiarity_rate'].mean():.2f}%")
        print(f"    Median familiarity: {notable['familiarity_rate'].median():.2f}%")
        print(f"    Range: {notable['familiarity_rate'].min():.2f}% - {notable['familiarity_rate'].max():.2f}%")
        
        print(f"\n  State prosecutors (n={len(state)}):")
        print(f"    Mean familiarity: {state['familiarity_rate'].mean():.2f}%")
        print(f"    Median familiarity: {state['familiarity_rate'].median():.2f}%")
        print(f"    Range: {state['familiarity_rate'].min():.2f}% - {state['familiarity_rate'].max():.2f}%")
        
        diff = notable['familiarity_rate'].mean() - state['familiarity_rate'].mean()
        print(f"\n  Difference: {diff:+.2f} percentage points")
        
        if diff > 0:
            print(f"  Notable prosecutors are MORE familiar (as expected for national figures)")
        else:
            print(f"  ⚠️ State prosecutors are MORE familiar (unexpected!)")
        
        # Top prosecutors by familiarity
        top_10 = self.df_prosecutors.nlargest(10, 'familiarity_rate')
        print("\nTOP 10 PROSECUTORS BY FAMILIARITY:")
        for idx, row in top_10.iterrows():
            notable_flag = "NOTABLE" if row['is_notable'] else "STATE  "
            print(f"  {notable_flag} - {row['name']:35s} - {row['familiarity_rate']:5.1f}% ({row['location']})")
        
        # Statistical test
        t_stat, p_val = ttest_ind(notable['familiarity_rate'], state['familiarity_rate'])
        print(f"\nT-test (Notable vs State): t={t_stat:.3f}, p={p_val:.4f}")
        if p_val < 0.05:
            print("  *** SIGNIFICANT DIFFERENCE ***")
        
        self.results['familiarity_comparison'] = {
            'notable_mean': notable['familiarity_rate'].mean(),
            'state_mean': state['familiarity_rate'].mean(),
            'difference': diff,
            't_stat': t_stat,
            'p_val': p_val,
            'top_10': top_10
        }
        
        return self
    
    def analyze_incumbency(self):
        """Analyze incumbency patterns (uses FULL matched sample)"""
        print_section("ANALYZING INCUMBENCY PATTERNS")

        if not hasattr(self, 'df_matched_all'):
            print("⚠ Election data not matched yet")
            return self

        df = self.df_matched_all[self.df_matched_all['num_elections'] > 0]

        incumbents = df[df['ever_ran_as_incumbent']]
        challengers = df[df['ever_ran_as_challenger']]

        print("INCUMBENCY VS CHALLENGER:")
        print(f"  Ever ran as incumbent: {len(incumbents)} prosecutors")
        print(f"    Mean familiarity: {incumbents['familiarity_rate'].mean():.2f}%")
        print(f"  Ever ran as challenger: {len(challengers)} prosecutors")
        print(f"    Mean familiarity: {challengers['familiarity_rate'].mean():.2f}%")

        inc_mean = incumbents['familiarity_rate'].mean() if len(incumbents) > 0 else np.nan
        chal_mean = challengers['familiarity_rate'].mean() if len(challengers) > 0 else np.nan
        diff = inc_mean - chal_mean if pd.notna(inc_mean) and pd.notna(chal_mean) else np.nan
        t_stat = p_val = np.nan
        if len(incumbents) > 0 and len(challengers) > 0:
            t_stat, p_val = ttest_ind(incumbents['familiarity_rate'],
                                     challengers['familiarity_rate'])
            print(f"  T-test: t={t_stat:.3f}, p={p_val:.4f}")
            if p_val < 0.05:
                print("  *** SIGNIFICANT DIFFERENCE ***")

        self.results['incumbency_analysis'] = {
            'incumbent_mean': inc_mean,
            'challenger_mean': chal_mean,
            'difference': diff,
            'incumbent_count': len(incumbents),
            'challenger_count': len(challengers),
            't_stat': t_stat,
            'p_val': p_val
        }

        return self

    def analyze_visibility_and_competition(self):
        """Extended analyses requested for visibility, contestation, and ideology."""

        print_section("EXTENDED VISIBILITY & COMPETITION ANALYSES")

        results = {}

        # ------------------------------------------------------------------
        # Bivariate familiarity vs ideology pattern
        # ------------------------------------------------------------------
        df_ideology = getattr(self, 'df_matched', pd.DataFrame()).copy()
        if not df_ideology.empty:
            df_ideology = df_ideology[pd.notna(df_ideology['mean_score']) & pd.notna(df_ideology['familiarity_rate'])]
            high = df_ideology[df_ideology['familiarity_rate'] > 30]
            low = df_ideology[df_ideology['familiarity_rate'] < 10]

            if len(high) > 1 and len(low) > 1:
                high_mean = high['mean_score'].mean()
                low_mean = low['mean_score'].mean()
                diff = high_mean - low_mean
                t_stat, p_val = stats.ttest_ind(high['mean_score'], low['mean_score'], equal_var=False, nan_policy='omit')
            else:
                high_mean = low_mean = diff = t_stat = p_val = np.nan

            results['bivariate_pattern'] = {
                'high_count': len(high),
                'high_mean': high_mean,
                'low_count': len(low),
                'low_mean': low_mean,
                'difference': diff,
                't_stat': t_stat,
                'p_val': p_val
            }

            if len(df_ideology) > 2:
                corr_r, corr_p = pearsonr(df_ideology['mean_score'], df_ideology['familiarity_rate'])
            else:
                corr_r = corr_p = np.nan

            results['familiarity_progressiveness_corr'] = {
                'sample_size': len(df_ideology),
                'pearson_r': corr_r,
                'pearson_p': corr_p
            }
        else:
            results['bivariate_pattern'] = None
            results['familiarity_progressiveness_corr'] = None

        # ------------------------------------------------------------------
        # National notable prosecutors: contested vs uncontested general elections
        # ------------------------------------------------------------------
        df_notable = getattr(self, 'df_matched_all', pd.DataFrame()).copy()
        if not df_notable.empty:
            df_notable = df_notable[df_notable['is_notable'] & (df_notable['num_elections'] > 0)]
            contested = df_notable[df_notable['ever_contested_general'] == True]
            uncontested = df_notable[df_notable['ever_contested_general'] == False]

            if len(contested) > 1 and len(uncontested) > 1:
                t_stat, p_val = stats.ttest_ind(
                    contested['familiarity_rate'], uncontested['familiarity_rate'],
                    equal_var=False, nan_policy='omit'
                )
                diff = contested['familiarity_rate'].mean() - uncontested['familiarity_rate'].mean()
            else:
                t_stat = p_val = diff = np.nan

            results['national_contestation'] = {
                'total_with_data': len(df_notable),
                'contested_count': len(contested),
                'contested_mean': contested['familiarity_rate'].mean() if len(contested) else np.nan,
                'uncontested_count': len(uncontested),
                'uncontested_mean': uncontested['familiarity_rate'].mean() if len(uncontested) else np.nan,
                'difference': diff,
                't_stat': t_stat,
                'p_val': p_val
            }
        else:
            results['national_contestation'] = None

        # ------------------------------------------------------------------
        # State-level prosecutors: contested vs uncontested, primary competition
        # ------------------------------------------------------------------
        df_state = getattr(self, 'df_matched_all', pd.DataFrame()).copy()
        if not df_state.empty:
            df_state = df_state[(~df_state['is_notable']) & (df_state['num_elections'] > 0)]
            contested = df_state[df_state['ever_contested_general'] == True]
            uncontested = df_state[df_state['ever_contested_general'] == False]

            if len(contested) > 1 and len(uncontested) > 1:
                state_t, state_p = stats.ttest_ind(
                    contested['familiarity_rate'], uncontested['familiarity_rate'],
                    equal_var=False, nan_policy='omit'
                )
                state_diff = contested['familiarity_rate'].mean() - uncontested['familiarity_rate'].mean()
            else:
                state_t = state_p = state_diff = np.nan

            close_primary = df_state[df_state['had_close_primary'] == True]
            not_close_primary = df_state[df_state['had_close_primary'] == False]
            if len(close_primary) > 1 and len(not_close_primary) > 1:
                primary_t, primary_p = stats.ttest_ind(
                    close_primary['familiarity_rate'], not_close_primary['familiarity_rate'],
                    equal_var=False, nan_policy='omit'
                )
                primary_diff = close_primary['familiarity_rate'].mean() - not_close_primary['familiarity_rate'].mean()
            else:
                primary_t = primary_p = primary_diff = np.nan

            margin_df = df_state[pd.notna(df_state['closest_primary_margin']) & pd.notna(df_state['familiarity_rate'])]
            if len(margin_df) > 1:
                corr_r, corr_p = pearsonr(margin_df['closest_primary_margin'], margin_df['familiarity_rate'])
            else:
                corr_r = corr_p = np.nan

            results['state_contestation'] = {
                'total_with_data': len(df_state),
                'contested_count': len(contested),
                'contested_mean': contested['familiarity_rate'].mean() if len(contested) else np.nan,
                'uncontested_count': len(uncontested),
                'uncontested_mean': uncontested['familiarity_rate'].mean() if len(uncontested) else np.nan,
                'difference': state_diff,
                't_stat': state_t,
                'p_val': state_p,
                'close_primary_count': len(close_primary),
                'not_close_primary_count': len(not_close_primary),
                'close_primary_mean': close_primary['familiarity_rate'].mean() if len(close_primary) else np.nan,
                'not_close_primary_mean': not_close_primary['familiarity_rate'].mean() if len(not_close_primary) else np.nan,
                'close_primary_diff': primary_diff,
                'close_primary_t': primary_t,
                'close_primary_p': primary_p,
                'primary_margin_corr': corr_r,
                'primary_margin_p': corr_p
            }
        else:
            results['state_contestation'] = None

        # ------------------------------------------------------------------
        # Logistic regression predicting close general victories
        # ------------------------------------------------------------------
        df_logit = getattr(self, 'df_matched_all', pd.DataFrame()).copy()
        logistic_summary = None
        if not df_logit.empty and 'closest_general_margin' in df_logit.columns:
            df_logit = df_logit[pd.notna(df_logit['closest_general_margin'])]
            if not df_logit.empty:
                df_logit = df_logit.copy()
                df_logit['close_victory'] = df_logit['closest_general_margin'] < Config.CLOSE_MARGIN_THRESHOLD
                df_logit['ever_ran_as_challenger'] = df_logit['ever_ran_as_challenger'].fillna(False).astype(int)
                df_logit['ever_contested_primary'] = df_logit['ever_contested_primary'].fillna(False).astype(int)
                df_logit['close_victory'] = df_logit['close_victory'].astype(int)

                logit_vars = df_logit[['close_victory', 'ever_ran_as_challenger', 'mean_score', 'ever_contested_primary']].dropna()
                if len(logit_vars) >= 10 and logit_vars['close_victory'].nunique() == 2:
                    try:
                        model = smf.logit(
                            'close_victory ~ ever_ran_as_challenger + mean_score + ever_contested_primary',
                            data=logit_vars
                        ).fit(disp=False)
                        logistic_summary = model
                    except Exception as exc:
                        print(f"⚠ Logistic regression failed: {exc}")

                logistic_results = {
                    'sample_size': len(logit_vars),
                    'close_victories': int(logit_vars['close_victory'].sum()),
                    'comfortable_victories': int(len(logit_vars) - logit_vars['close_victory'].sum()),
                    'close_share': logit_vars['close_victory'].mean() if len(logit_vars) else np.nan,
                    'model': logistic_summary
                }
            else:
                logistic_results = {
                    'sample_size': 0,
                    'close_victories': 0,
                    'comfortable_victories': 0,
                    'close_share': np.nan,
                    'model': None
                }
        else:
            logistic_results = None

        results['logistic_close_victory'] = logistic_results

        # ------------------------------------------------------------------
        # Pathway to vulnerability: challenger entry and margins
        # ------------------------------------------------------------------
        if not df_ideology.empty:
            progressive_mask = df_ideology['mean_score'] >= 3.0
            non_progressive_mask = df_ideology['mean_score'] < 3.0

            progressive_df = df_ideology[progressive_mask]
            non_progressive_df = df_ideology[non_progressive_mask]

            challenger_rate_prog = progressive_df['ever_ran_as_challenger'].mean() if len(progressive_df) else np.nan
            challenger_rate_non = non_progressive_df['ever_ran_as_challenger'].mean() if len(non_progressive_df) else np.nan

            challenger_margins = df_ideology[df_ideology['ever_ran_as_challenger'] == True]
            non_challenger_margins = df_ideology[df_ideology['ever_ran_as_challenger'] == False]
            if len(challenger_margins) > 1 and len(non_challenger_margins) > 1:
                margin_diff = challenger_margins['closest_general_margin'].mean() - non_challenger_margins['closest_general_margin'].mean()
            else:
                margin_diff = np.nan

            try:
                challenger_model = smf.logit(
                    'ever_ran_as_challenger ~ mean_score',
                    data=df_ideology.dropna(subset=['mean_score', 'ever_ran_as_challenger'])
                ).fit(disp=False)
            except Exception:
                challenger_model = None

            results['vulnerability_pathway'] = {
                'progressive_challenger_rate': challenger_rate_prog,
                'non_progressive_challenger_rate': challenger_rate_non,
                'challenger_margin_difference': margin_diff,
                'challenger_model': challenger_model
            }
        else:
            results['vulnerability_pathway'] = None

        # ------------------------------------------------------------------
        # Contestation rates by ideology category
        # ------------------------------------------------------------------
        if not df_ideology.empty and 'ideology_category' in df_ideology.columns:
            ideology_counts = df_ideology['ideology_category'].value_counts().to_dict()
            comp_primary = df_ideology.groupby('ideology_category')['ever_contested_primary'].mean()
            comp_general = df_ideology.groupby('ideology_category')['ever_contested_general'].mean()

            subset = df_ideology[df_ideology['ideology_category'].isin(['Progressive', 'Traditional'])]
            if not subset.empty:
                chi_primary = chi2_contingency(pd.crosstab(
                    subset['ideology_category'], subset['ever_contested_primary']
                )) if subset['ever_contested_primary'].nunique() > 1 else (np.nan, np.nan, np.nan, None)
                chi_general = chi2_contingency(pd.crosstab(
                    subset['ideology_category'], subset['ever_contested_general']
                )) if subset['ever_contested_general'].nunique() > 1 else (np.nan, np.nan, np.nan, None)
            else:
                chi_primary = chi_general = (np.nan, np.nan, np.nan, None)

            results['contestation_by_ideology'] = {
                'counts': ideology_counts,
                'primary_rates': comp_primary.to_dict(),
                'general_rates': comp_general.to_dict(),
                'chi_primary': chi_primary,
                'chi_general': chi_general
            }
        else:
            results['contestation_by_ideology'] = None

        # ------------------------------------------------------------------
        # Mean ideology by rater position
        # ------------------------------------------------------------------
        survey = getattr(self, 'df_survey', pd.DataFrame())
        rating_cols = [c for c in (self.notable_cols + self.state_da_cols) if c in survey.columns]
        rater_results = None
        if not survey.empty and rating_cols:
            working = survey.copy()
            working['_respondent_id'] = working.get('ResponseId', working.index)
            id_vars = ['_respondent_id']
            if 'position' in working.columns:
                id_vars.append('position')
            else:
                working['position'] = ''
                id_vars.append('position')

            melted = working[id_vars + rating_cols].melt(id_vars=id_vars, value_vars=rating_cols,
                                                          var_name='column', value_name='rating')
            melted['score'] = melted['rating'].map(Config.RATING_MAP)
            melted = melted.dropna(subset=['score'])
            if not melted.empty:
                meta = self.df_prosecutors[['column', 'name']]
                melted = melted.merge(meta, how='left', left_on='column', right_on='column')

                def classify_position(value):
                    text = str(value).strip()
                    if not text or text.lower().startswith('your responses to this survey'):
                        return 'Other professionals'
                    text_lower = text.lower()
                    if 'prosecutor' in text_lower:
                        return 'Prosecutors rating prosecutors'
                    if 'academic' in text_lower:
                        return 'Academics rating prosecutors'
                    if 'defense attorney' in text_lower:
                        return 'Defense attorneys rating prosecutors'
                    return 'Other professionals'

                melted['position_group'] = melted['position'].apply(classify_position)
                rater_group = melted.groupby('position_group')['score']
                rater_means = rater_group.mean().to_dict()
                counts = rater_group.count().to_dict()

                groups_for_anova = [grp['score'].values for _, grp in melted.groupby('position_group') if len(grp) > 1]
                if len(groups_for_anova) >= 2:
                    f_stat, f_p = stats.f_oneway(*groups_for_anova)
                    dof1 = len(groups_for_anova) - 1
                    dof2 = sum(len(g) for g in groups_for_anova) - len(groups_for_anova)
                else:
                    f_stat = f_p = np.nan
                    dof1 = dof2 = np.nan

                rater_results = {
                    'means': rater_means,
                    'counts': counts,
                    'anova_f': f_stat,
                    'anova_p': f_p,
                    'anova_df1': dof1,
                    'anova_df2': dof2
                }

        results['rater_position_scores'] = rater_results

        # ------------------------------------------------------------------
        # Regional patterns
        # ------------------------------------------------------------------
        region_df = df_ideology.copy()
        if not region_df.empty:
            region_df['region'] = region_df['state_name'].apply(lambda x: self._assign_region(x))
            region_group = region_df.groupby('region')['mean_score']
            region_means = region_group.mean().to_dict()
            groups = [grp['mean_score'].dropna().values for _, grp in region_df.groupby('region') if len(grp.dropna(subset=['mean_score'])) > 1]
            if len(groups) >= 2:
                region_f, region_p = stats.f_oneway(*groups)
                df1 = len(groups) - 1
                df2 = sum(len(g) for g in groups) - len(groups)
            else:
                region_f = region_p = np.nan
                df1 = df2 = np.nan

            results['regional_patterns'] = {
                'means': region_means,
                'anova_f': region_f,
                'anova_p': region_p,
                'anova_df1': df1,
                'anova_df2': df2
            }
        else:
            results['regional_patterns'] = None

        # ------------------------------------------------------------------
        # Urban vs rural (notable prosecutors only)
        # ------------------------------------------------------------------
        notable_pop = df_ideology[df_ideology['is_notable'] == True].copy()
        if not notable_pop.empty:
            pop_group = notable_pop.groupby('population_category')['mean_score']
            pop_means = pop_group.mean().to_dict()
            pop_counts = pop_group.count().to_dict()
            corr_df = notable_pop[pd.notna(notable_pop['county_population']) & pd.notna(notable_pop['mean_score'])]
            if len(corr_df) > 1:
                pop_corr, pop_corr_p = pearsonr(corr_df['county_population'], corr_df['mean_score'])
            else:
                pop_corr = pop_corr_p = np.nan

            results['urban_rural'] = {
                'means': pop_means,
                'counts': pop_counts,
                'population_correlation': pop_corr,
                'population_corr_p': pop_corr_p
            }
        else:
            results['urban_rural'] = None

        # ------------------------------------------------------------------
        # Geographic concentration of progressive vs traditional prosecutors
        # ------------------------------------------------------------------
        if not df_ideology.empty:
            progressive = df_ideology[df_ideology['mean_score'] >= 2.5]
            traditional = df_ideology[df_ideology['mean_score'] < 2.5]

            top10_prog = progressive.sort_values('mean_score', ascending=False).head(10)
            california_top10 = top10_prog['state_name'].fillna('').str.contains('California').sum()
            major_city_top10 = top10_prog['population_category'].eq(Config.POPULATION_CATEGORY_LABELS['large']).sum()
            biden_prog = progressive['state_name'].apply(lambda s: s in Config.BIDEN_STATES_2020).sum()

            traditional_states = traditional['state_name'].apply(lambda s: s if isinstance(s, str) else '').tolist()
            southern_states = [s for s in traditional_states if s in {'Alabama', 'Arkansas', 'Florida', 'Georgia', 'Louisiana', 'Mississippi', 'South Carolina', 'Tennessee', 'Texas'}]
            trump_traditional = sum(1 for s in traditional_states if s in Config.TRUMP_STATES_2020)

            results['geographic_concentration'] = {
                'progressive_total': len(progressive),
                'progressive_top10_ca': int(california_top10),
                'progressive_top10_major_city': int(major_city_top10),
                'progressive_biden_states': int(biden_prog),
                'traditional_total': len(traditional),
                'traditional_southern_states': len(southern_states),
                'traditional_trump_states': int(trump_traditional)
            }
        else:
            results['geographic_concentration'] = None

        self.results['visibility_competition'] = results
        return self
    
    def analyze_contestation(self):
        """Analyze contestation patterns (uses FULL matched sample for broad patterns)"""
        print_section("ANALYZING CONTESTATION PATTERNS")
        
        if not hasattr(self, 'df_matched_all'):
            print("⚠ Election data not matched yet")
            return self
        
        df = self.df_matched_all[self.df_matched_all['num_elections'] > 0]
        
        # Overall contestation rates
        print("OVERALL CONTESTATION RATES:")
        print(f"  Ever contested primary: {df['ever_contested_primary'].mean()*100:.1f}%")
        print(f"  Ever contested general: {df['ever_contested_general'].mean()*100:.1f}%")
        print(f"  Ever contested any: {df['ever_contested_any'].mean()*100:.1f}%")
        
        # Familiarity by contestation
        print("\nFAMILIARITY BY CONTESTATION:")
        
        contested = df[df['ever_contested_any']]
        uncontested = df[~df['ever_contested_any']]

        print(f"  Ever contested (n={len(contested)}): {contested['familiarity_rate'].mean():.2f}%")
        print(f"  Never contested (n={len(uncontested)}): {uncontested['familiarity_rate'].mean():.2f}%")

        t_stat = p_val = np.nan
        if len(contested) > 0 and len(uncontested) > 0:
            t_stat, p_val = ttest_ind(contested['familiarity_rate'],
                                     uncontested['familiarity_rate'])
            print(f"  T-test: t={t_stat:.3f}, p={p_val:.4f}")
            if p_val < 0.05:
                print("  *** SIGNIFICANT DIFFERENCE ***")

        self.results['contestation_analysis'] = {
            'contested_mean': contested['familiarity_rate'].mean() if len(contested) > 0 else np.nan,
            'uncontested_mean': uncontested['familiarity_rate'].mean() if len(uncontested) > 0 else np.nan,
            'difference': (contested['familiarity_rate'].mean() - uncontested['familiarity_rate'].mean())
                          if len(contested) > 0 and len(uncontested) > 0 else np.nan,
            'contested_count': len(contested),
            'uncontested_count': len(uncontested),
            't_stat': t_stat,
            'p_val': p_val
        }

        return self
    
    def analyze_recall_risk(self):
        """Analyze recall risk patterns (uses FILTERED sample for ideology scores)"""
        print_section("ANALYZING RECALL RISK")
        
        if not hasattr(self, 'df_matched'):
            print("⚠ Election data not matched yet")
            return self
        
        recalled = self.df_matched[self.df_matched['recalled']]
        not_recalled = self.df_matched[~self.df_matched['recalled'] & self.df_matched['is_notable']]
        
        if len(recalled) == 0:
            print("⚠ No recalled prosecutors in matched data")
            return self
        
        print("RECALLED PROSECUTORS:")
        for idx, row in recalled.iterrows():
            print(f"\n  {row['name']}:")
            print(f"    Ideology score: {row['mean_score']:.2f}")
            if pd.notna(row.get('closest_general_margin')):
                print(f"    Closest general margin: {row['closest_general_margin']:.1f}%")
            print(f"    Familiarity: {row['familiarity_rate']:.1f}%")
        
        if len(not_recalled) > 0:
            print(f"\nNON-RECALLED NOTABLE PROSECUTORS (n={len(not_recalled)}):")
            print(f"  Mean general margin: {not_recalled['closest_general_margin'].mean():.1f}%")
            print(f"  Mean ideology score: {not_recalled['mean_score'].mean():.2f}")
            
            # Statistical test
            recalled_margins = recalled['closest_general_margin'].dropna()
            not_recalled_margins = not_recalled['closest_general_margin'].dropna()
            
            if len(recalled_margins) > 0 and len(not_recalled_margins) > 0:
                t_stat, p_val = ttest_ind(recalled_margins, not_recalled_margins)
                print(f"\nT-test on general margins: t={t_stat:.3f}, p={p_val:.4f}")
                if p_val < 0.05:
                    print("  *** SIGNIFICANT DIFFERENCE ***")
        
        return self
    
    def run_multivariate_models(self):
        """Run TWO multivariate regression models with PROPER sample selection"""
        print_section("MULTIVARIATE REGRESSION MODELS")
        
        # ========================================================================
        # MODEL 1: Electoral Factors Only - Uses FULL matched sample
        # ========================================================================
        
        print("MODEL 1: FAMILIARITY DRIVERS - Electoral Factors Only")
        print("Sample: FULL matched sample (all prosecutors with election data)")
        print("Purpose: Proper state vs notable comparison")
        print()
        
        if not hasattr(self, 'df_matched_all'):
            print("⚠ Election data not matched yet")
            return self
        
        df1 = self.df_matched_all[self.df_matched_all['num_elections'] > 0].copy()
        
        print(f"Sample size: {len(df1)} prosecutors")
        print(f"  Notable: {df1['is_notable'].sum()}")
        print(f"  State: {(~df1['is_notable']).sum()}")
        print()
        
        # Prepare variables for Model 1 (no ideology - doesn't need ≥10 ratings)
        X1 = pd.DataFrame({
            'ever_contested_general': df1['ever_contested_general'].astype(int),
            'ever_ran_as_challenger': df1['ever_ran_as_challenger'].astype(int),
            'num_elections': df1['num_elections'],
            'is_notable': df1['is_notable'].astype(int),
            'ever_ran_as_incumbent': df1['ever_ran_as_incumbent'].astype(int)
        })
        
        X1 = sm.add_constant(X1)
        y1 = df1['familiarity_rate']
        
        # Run Model 1
        model1 = sm.OLS(y1, X1).fit()
        
        print(model1.summary())
        print()
        
        print("KEY FINDINGS (Model 1):")
        for var in ['ever_contested_general', 'ever_ran_as_challenger', 
                   'ever_ran_as_incumbent', 'num_elections', 'is_notable']:
            coef = model1.params[var]
            pval = model1.pvalues[var]
            sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
            print(f"  {var:30s}: β={coef:7.2f}  p={pval:.4f} {sig}")
        
        print(f"\nR²: {model1.rsquared:.3f}")
        print(f"Adjusted R²: {model1.rsquared_adj:.3f}")
        
        self.results['familiarity_model_electoral'] = model1
        
        # ========================================================================
        # MODEL 2: Add Ideology - Uses FILTERED matched sample
        # ========================================================================
        
        print("\n" + "="*80)
        print("MODEL 2: FAMILIARITY DRIVERS - Electoral Factors + Ideology")
        print("Sample: FILTERED matched sample (prosecutors with ≥10 ratings)")
        print("Purpose: Test ideology effects (sample heavily skewed toward notable)")
        print()
        
        df2 = self.df_matched[self.df_matched['num_elections'] > 0].copy()
        
        print(f"Sample size: {len(df2)} prosecutors")
        print(f"  Notable: {df2['is_notable'].sum()} ({100*df2['is_notable'].mean():.1f}%)")
        print(f"  State: {(~df2['is_notable']).sum()} ({100*(~df2['is_notable']).mean():.1f}%)")
        print()
        
        # Prepare variables for Model 2 (with ideology)
        X2 = pd.DataFrame({
            'mean_score': df2['mean_score'],
            'ever_contested_general': df2['ever_contested_general'].astype(int),
            'ever_ran_as_challenger': df2['ever_ran_as_challenger'].astype(int),
            'num_elections': df2['num_elections'],
            'is_notable': df2['is_notable'].astype(int),
            'ever_ran_as_incumbent': df2['ever_ran_as_incumbent'].astype(int)
        })
        
        X2 = sm.add_constant(X2)
        y2 = df2['familiarity_rate']
        
        # Run Model 2
        model2 = sm.OLS(y2, X2).fit()
        
        print(model2.summary())
        print()
        
        print("KEY FINDINGS (Model 2):")
        for var in ['mean_score', 'ever_contested_general', 'ever_ran_as_challenger', 
                   'ever_ran_as_incumbent', 'num_elections', 'is_notable']:
            coef = model2.params[var]
            pval = model2.pvalues[var]
            sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
            print(f"  {var:30s}: β={coef:7.2f}  p={pval:.4f} {sig}")
        
        print(f"\nR²: {model2.rsquared:.3f}")
        print(f"Adjusted R²: {model2.rsquared_adj:.3f}")
        
        self.results['familiarity_model_with_ideology'] = model2
        
        # ========================================================================
        # INTERPRETATION
        # ========================================================================
        
        print("\n" + "="*80)
        print("INTERPRETATION:")
        print("="*80)
        print()
        print(f"Model 1 provides proper estimate of state vs notable difference (n={len(df1)}):")
        print(f"  is_notable coefficient: {model1.params['is_notable']:.2f} (p={model1.pvalues['is_notable']:.4f})")
        if model1.params['is_notable'] < 0:
            print(f"  → State prosecutors have {abs(model1.params['is_notable']):.1f} percentage points HIGHER familiarity")
        print()
        print(f"Model 2 shows ideology effects within mostly notable sample (n={len(df2)}):")
        print(f"  mean_score coefficient: {model2.params['mean_score']:.2f} (p={model2.pvalues['mean_score']:.4f})")
        print(f"  is_notable coefficient: {model2.params['is_notable']:.2f} (unreliable - only {(~df2['is_notable']).sum()} state prosecutors)")
        print()
        
        return self
    
    def export_results(self, output_dir):
        """Export all results to CSV files"""
        print_section("EXPORTING RESULTS")
        
        ensure_dir(output_dir)
        
        # Export all prosecutors
        self.df_prosecutors.to_csv(f'{output_dir}/all_prosecutors_CORRECTED.csv', index=False)
        print(f"✓ Exported all_prosecutors_CORRECTED.csv ({len(self.df_prosecutors)} prosecutors)")
        
        # Export filtered prosecutors
        if hasattr(self, 'df_prosecutors_filtered_ideology'):
            self.df_prosecutors_filtered_ideology.to_csv(
                f'{output_dir}/prosecutors_filtered_ideology_CORRECTED.csv', index=False)
            print(f"✓ Exported prosecutors_filtered_ideology_CORRECTED.csv ({len(self.df_prosecutors_filtered_ideology)} prosecutors)")
        
        # Export matched samples
        if hasattr(self, 'df_matched_all'):
            self.df_matched_all.to_csv(f'{output_dir}/matched_prosecutors_FULL_CORRECTED.csv', index=False)
            print(f"✓ Exported matched_prosecutors_FULL_CORRECTED.csv ({len(self.df_matched_all)} prosecutors)")
            print(f"  Use this for: Familiarity analyses, electoral patterns")
        
        if hasattr(self, 'df_matched'):
            self.df_matched.to_csv(f'{output_dir}/matched_prosecutors_FILTERED_CORRECTED.csv', index=False)
            print(f"✓ Exported matched_prosecutors_FILTERED_CORRECTED.csv ({len(self.df_matched)} prosecutors)")
            print(f"  Use this for: Ideology analyses, progressive vs traditional comparisons")
        
        # Export summary report
        self.export_summary_report(output_dir)
        
        return self
    
    def export_summary_report(self, output_dir):
        """Export a summary report of key findings"""
        report_path = f'{output_dir}/CORRECTED_FINDINGS_SUMMARY.txt'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("CORRECTED PROSECUTOR ANALYSIS - KEY FINDINGS SUMMARY\n")
            f.write("="*80 + "\n\n")
            
            f.write("CRITICAL CORRECTIONS APPLIED:\n")
            f.write("-" * 80 + "\n")
            f.write("1. Notable prosecutor familiarity = substantive_ratings / 407 (completed national section)\n")
            f.write("2. State prosecutor familiarity = substantive_ratings / state_respondents_who_engaged\n")
            f.write("3. Scale corrected to 1-4 (Very Traditional to Very Progressive)\n")
            f.write("4. TWO regression models: Model 1 (electoral, n=191) & Model 2 (with ideology, n=51)\n\n")
            f.write("METHODOLOGY:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total initiated: 496\n")
            f.write(f"After consent: 493\n")
            f.write(f"Completed national prosecutor section: 407 (82.1%)\n")
            f.write(f"Completed full survey: 365 (74.0%)\n\n")
            
            if 'familiarity_comparison' in self.results:
                fam = self.results['familiarity_comparison']
                f.write("FAMILIARITY RATES (CORRECTED):\n")
                f.write("-" * 80 + "\n")
                f.write(f"Notable prosecutors: {fam['notable_mean']:.2f}% mean familiarity\n")
                f.write(f"State prosecutors: {fam['state_mean']:.2f}% mean familiarity\n")
                f.write(f"Difference: {fam['difference']:+.2f} percentage points\n")
                f.write(f"T-test: t={fam['t_stat']:.3f}, p={fam['p_val']:.4f}\n\n")
            
            if 'familiarity_model_electoral' in self.results:
                model1 = self.results['familiarity_model_electoral']
                f.write("MODEL 1: FAMILIARITY DRIVERS (Electoral Factors, n=191):\n")
                f.write("-" * 80 + "\n")
                f.write(f"R² = {model1.rsquared:.3f}\n")
                f.write(f"State vs Notable: β={model1.params['is_notable']:.2f}, p={model1.pvalues['is_notable']:.4f}\n")
                f.write(f"Ran as Challenger: β={model1.params['ever_ran_as_challenger']:.2f}, p={model1.pvalues['ever_ran_as_challenger']:.4f}\n")
                f.write(f"Contested General: β={model1.params['ever_contested_general']:.2f}, p={model1.pvalues['ever_contested_general']:.4f}\n")
                f.write(f"Number of Elections: β={model1.params['num_elections']:.2f}, p={model1.pvalues['num_elections']:.4f}\n\n")
            
            if 'familiarity_model_with_ideology' in self.results:
                model2 = self.results['familiarity_model_with_ideology']
                f.write("MODEL 2: FAMILIARITY + IDEOLOGY (n=51, 94% notable):\n")
                f.write("-" * 80 + "\n")
                f.write(f"R² = {model2.rsquared:.3f}\n")
                f.write(f"Progressiveness: β={model2.params['mean_score']:.2f}, p={model2.pvalues['mean_score']:.4f}\n")
                f.write(f"Contested General: β={model2.params['ever_contested_general']:.2f}, p={model2.pvalues['ever_contested_general']:.4f}\n")
                f.write("Note: is_notable coefficient unreliable due to only 3 state prosecutors\n\n")
            
            if 'top_10' in self.results.get('familiarity_comparison', {}):
                f.write("TOP 10 PROSECUTORS BY FAMILIARITY:\n")
                f.write("-" * 80 + "\n")
                for idx, row in self.results['familiarity_comparison']['top_10'].iterrows():
                    f.write(f"  {row['name']:35s} - {row['familiarity_rate']:5.1f}%\n")
                f.write("\n")
            
            f.write("\nAll detailed results available in CSV files.\n")
        
        print(f"✓ Exported summary report: {report_path}")



class IntegratedReporter:
    """Generate extended outputs that combine the strongest portions of prior scripts."""

    def __init__(self, analyzer, config=Config):
        self.analyzer = analyzer
        self.config = config
        ensure_dir(config.OUTPUT_DIR)
        ensure_dir(config.VIZ_DIR)

    def _savefig(self, path):
        plt.tight_layout()
        plt.savefig(path, dpi=200, bbox_inches='tight')
        plt.close()

    def _prepare_notable_df(self):
        df = self.analyzer.df_prosecutors.copy()
        if df.empty:
            return df
        df = df[df['is_notable']].copy()
        if df.empty:
            return df
        df['Name'] = df['name']
        df['Location'] = df['location']
        df['Mean_Score'] = df['mean_score']
        df['Median_Score'] = df['median_score']
        df['Std_Dev'] = df['std_score']
        df['Substantive_Ratings'] = df['substantive_ratings']
        df['Familiarity_Rate'] = df['familiarity_rate']
        df['Academic_Mean'] = df['academic_mean']
        df['Defense_Mean'] = df['defense_mean']
        df['Prosecutor_Mean'] = df['prosecutor_mean']
        df['Very_Progressive_Pct'] = df['very_progressive_pct']
        df['Very_Traditional_Pct'] = df['very_traditional_pct']
        df['Progressive_Pct'] = df['progressive_pct']
        df['Traditional_Pct'] = df['traditional_pct']
        df['Column'] = df['column']
        return df

    def _prepare_state_df(self):
        state_df = getattr(self.analyzer, 'state_df', pd.DataFrame()).copy()
        return state_df if state_df is not None else pd.DataFrame()

    def generate_visualizations(self):
        print_section("GENERATING EXTENDED VISUALIZATIONS")
        df = self.analyzer.df_prosecutors.copy()

        try:
            notable = df[df['is_notable']]
            state = df[~df['is_notable']]
            plt.figure(figsize=(5, 4))
            plt.bar(['Notables', 'State'], [notable['familiarity_rate'].mean(), state['familiarity_rate'].mean()],
                    color=['#1f77b4', '#ff7f0e'])
            plt.ylabel('Mean familiarity (%)')
            plt.title('Familiarity: Notables vs State')
            self._savefig(Path(self.config.VIZ_DIR) / 'fig1_familiarity_notables_vs_state.png')
            print("✓ Familiarity comparison figure saved")
        except Exception as exc:
            print(f"[Viz] Familiarity comparison failed: {exc}")

        try:
            notables = df[df['is_notable'] & pd.notna(df['mean_score'])]
            if len(notables) >= 10:
                top10 = notables.sort_values('mean_score', ascending=False).head(10)
                bot10 = notables.sort_values('mean_score', ascending=True).head(10)
                for subset, title, fname in [
                    (top10.iloc[::-1], 'Top 10 Notables (Most Progressive)', 'fig2a_top10_notables.png'),
                    (bot10.iloc[::-1], 'Bottom 10 Notables (Most Traditional)', 'fig2b_bottom10_notables.png'),
                ]:
                    plt.figure(figsize=(7, 5))
                    y = np.arange(len(subset))
                    plt.barh(y, subset['mean_score'], color='#2ca02c')
                    plt.yticks(y, subset['name'])
                    plt.xlabel('Mean ideology (1=Trad → 4=Prog)')
                    plt.title(title)
                    self._savefig(Path(self.config.VIZ_DIR) / fname)
                print("✓ Ideology ranking figures saved")
        except Exception as exc:
            print(f"[Viz] Ideology ranking failed: {exc}")

        try:
            dfm = self.analyzer.df_matched_all.copy()
            dfm = dfm[pd.notna(dfm['closest_general_margin']) & pd.notna(dfm['familiarity_rate'])]
            if len(dfm) >= 3:
                x = dfm['closest_general_margin'].values
                y = dfm['familiarity_rate'].values
                A = np.vstack([x, np.ones_like(x)]).T
                coef, *_ = np.linalg.lstsq(A, y, rcond=None)
                xs = np.linspace(x.min(), x.max(), 100)
                ys = coef[0] * xs + coef[1]
                plt.figure(figsize=(6, 4))
                plt.scatter(x, y, alpha=0.7)
                plt.plot(xs, ys, color='black', linestyle='--')
                plt.xlabel('Closest general margin (%)')
                plt.ylabel('Familiarity (%)')
                plt.title('Familiarity vs Electoral Competitiveness')
                self._savefig(Path(self.config.VIZ_DIR) / 'fig3_familiarity_vs_margin.png')
                print("✓ Familiarity vs margin scatter saved")
        except Exception as exc:
            print(f"[Viz] Familiarity vs margin failed: {exc}")

        try:
            top6 = df[df['is_notable']].sort_values('familiarity_rate', ascending=False).head(6)
            if len(top6) > 0:
                plt.figure(figsize=(7, 4))
                x = np.arange(len(top6))
                means = top6['mean_score'].fillna(0).values
                sds = top6['std_score'].fillna(0).values if 'std_score' in top6 else np.zeros_like(means)
                plt.bar(x, means, yerr=sds, capsize=4, color='#9467bd')
                plt.xticks(x, top6['name'], rotation=30, ha='right')
                plt.ylabel('Mean ideology (1=Trad → 4=Prog)')
                plt.title('Ideology (mean ± sd): Top-6 Familiar Notables')
                self._savefig(Path(self.config.VIZ_DIR) / 'fig4_ideology_summary_top6.png')
                print("✓ Top familiar ideology summary saved")
        except Exception as exc:
            print(f"[Viz] Ideology summary failed: {exc}")

        try:
            dfm = self.analyzer.df_matched_all.copy()
            if len(dfm):
                dfm['bucket'] = np.where(
                    dfm['mean_score'] >= 3.0, 'Progressive-ish (≥3.0)',
                    np.where(dfm['mean_score'] <= 2.0, 'Traditional-ish (≤2.0)', 'Middle')
                )
                rates = dfm.groupby('bucket')['ever_contested_any'].mean().reindex(
                    ['Traditional-ish (≤2.0)', 'Middle', 'Progressive-ish (≥3.0)']
                )
                plt.figure(figsize=(6, 4))
                plt.bar(rates.index, rates.values * 100.0, color=['#8c564b', '#7f7f7f', '#17becf'])
                plt.ylabel('Ever contested (any) — %')
                plt.title('Contestation vs Ideology bucket')
                plt.xticks(rotation=20)
                self._savefig(Path(self.config.VIZ_DIR) / 'fig5_contestation_vs_ideology.png')
                print("✓ Contestation bucket figure saved")
        except Exception as exc:
            print(f"[Viz] Contestation bucket failed: {exc}")

        try:
            recalled = self.analyzer.df_matched_all.copy()
            recalled = recalled[recalled['recalled'] == True]
            if len(recalled) > 0:
                plt.figure(figsize=(6, 3.5))
                y = np.arange(len(recalled))
                plt.barh(y, recalled['closest_general_margin'].fillna(0), color='#d62728')
                plt.yticks(y, recalled['name'])
                plt.xlabel('Closest general election margin (%)')
                plt.title('Recall context: margins')
                self._savefig(Path(self.config.VIZ_DIR) / 'fig6_recalled_margins.png')
                print("✓ Recall context figure saved")
        except Exception as exc:
            print(f"[Viz] Recall context failed: {exc}")

        # Legacy visualization suite (top-to-bottom from prior scripts)
        self.generate_legacy_visualizations()
        self.generate_additional_rankings()
        self.generate_election_visualizations()
        self.generate_tier1_visualization()

        return self

    def generate_legacy_visualizations(self):
        notable_df = self._prepare_notable_df()
        state_df = self._prepare_state_df()
        viz_dir = Path(self.config.VIZ_DIR)

        if notable_df.empty:
            print("[Viz] No notable prosecutors available for legacy charts")
            return

        try:
            self._viz_rankings_top20(notable_df, viz_dir)
            print("✓ Legacy viz_1_rankings.png saved")
        except Exception as exc:
            print(f"[Viz] Legacy rankings failed: {exc}")

        try:
            self._viz_all_notables(notable_df, viz_dir)
            print("✓ Legacy viz_2_all_50_das.png saved")
        except Exception as exc:
            print(f"[Viz] Legacy all-notables failed: {exc}")

        try:
            self._viz_position_effects(notable_df, viz_dir)
            print("✓ Legacy viz_3_position_effects.png saved")
        except Exception as exc:
            print(f"[Viz] Position effects failed: {exc}")

        try:
            self._viz_familiarity_crisis(notable_df, viz_dir)
            print("✓ Legacy viz_4_familiarity.png saved")
        except Exception as exc:
            print(f"[Viz] Familiarity crisis failed: {exc}")

        try:
            if hasattr(self.analyzer, 'transitions_df') and not self.analyzer.transitions_df.empty:
                self._viz_transitions(self.analyzer.transitions_df, viz_dir)
                print("✓ Legacy viz_5_transitions.png saved")
        except Exception as exc:
            print(f"[Viz] Transitions failed: {exc}")

        try:
            self._viz_controversy(notable_df, viz_dir)
            print("✓ Legacy viz_6_controversy.png saved")
        except Exception as exc:
            print(f"[Viz] Controversy viz failed: {exc}")

        try:
            self._viz_geographic(notable_df, viz_dir)
            print("✓ Legacy viz_7_geographic.png saved")
        except Exception as exc:
            print(f"[Viz] Geographic viz failed: {exc}")

        try:
            if not state_df.empty:
                self._viz_national_vs_state(notable_df, state_df, viz_dir)
                print("✓ Legacy viz_8_national_vs_state.png saved")
        except Exception as exc:
            print(f"[Viz] National vs state failed: {exc}")

    def generate_additional_rankings(self):
        notable_df = self._prepare_notable_df()
        viz_dir = Path(self.config.VIZ_DIR)
        if notable_df.empty:
            return

        try:
            reliable = notable_df[notable_df['Substantive_Ratings'] >= 30].copy()
            if reliable.empty:
                return
            top10 = reliable.nlargest(10, 'Mean_Score')
            bottom10 = reliable.nsmallest(10, 'Mean_Score')

            plt.figure(figsize=(7, 5))
            plt.barh(top10['Name'], top10['Mean_Score'], color=plt.cm.RdYlGn((top10['Mean_Score'] - 1) / 3))
            plt.gca().invert_yaxis()
            plt.xlabel('Mean Ideology Score (4=Very Progressive)')
            plt.title('Top 10 Progressive (Notables)')
            self._savefig(viz_dir / 'rankings_top10_progressive.png')

            plt.figure(figsize=(7, 5))
            plt.barh(bottom10['Name'], bottom10['Mean_Score'], color=plt.cm.RdYlGn((bottom10['Mean_Score'] - 1) / 3))
            plt.gca().invert_yaxis()
            plt.xlabel('Mean Ideology Score (1=Very Traditional)')
            plt.title('Bottom 10 (Most Traditional) — Notables')
            self._savefig(viz_dir / 'rankings_bottom10_traditional.png')

            plt.figure(figsize=(6, 4))
            plt.hist(notable_df['Familiarity_Rate'].dropna(), bins=20, color='#1f77b4', alpha=0.8)
            plt.xlabel('Familiarity Rate (%)')
            plt.ylabel('Count of Notables')
            plt.title('Familiarity Distribution — Notables')
            self._savefig(viz_dir / 'familiarity_notables_hist.png')

            state_df = self.analyzer.df_prosecutors[~self.analyzer.df_prosecutors['is_notable']]
            if not state_df.empty:
                plt.figure(figsize=(6, 4))
                plt.hist(state_df['familiarity_rate'].dropna(), bins=20, color='#ff7f0e', alpha=0.8)
                plt.xlabel('Familiarity Rate (%)')
                plt.ylabel('Count of State DAs')
                plt.title('Familiarity Distribution — State DAs')
                self._savefig(viz_dir / 'familiarity_states_hist.png')

            transitions = getattr(self.analyzer, 'transitions_df', pd.DataFrame())
            if not transitions.empty:
                for _, row in transitions.iterrows():
                    plt.figure(figsize=(5, 4))
                    plt.plot([0, 1], [row['Pred_Score'], row['Succ_Score']], marker='o')
                    plt.xticks([0, 1], [row['Predecessor'], row['Successor']], rotation=0)
                    plt.ylabel('Mean Ideology Score (1–4)')
                    plt.title(f"Transition: {row['Predecessor']} → {row['Successor']}\n{row['Jurisdiction']}")
                    plt.ylim(1, 4)
                    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
                    fname = re.sub(r"[^a-z0-9]+", "_", row['Predecessor'].split()[-1].lower())
                    tname = re.sub(r"[^a-z0-9]+", "_", row['Successor'].split()[-1].lower())
                    self._savefig(viz_dir / f'transition_{fname}_to_{tname}.png')
        except Exception as exc:
            print(f"[Viz] Additional rankings failed: {exc}")

    def generate_election_visualizations(self):
        df = getattr(self.analyzer, 'df_matched_all', pd.DataFrame()).copy()
        if df.empty:
            return

        viz_dir = Path(self.config.VIZ_DIR)
        df['had_close_any'] = df['had_close_primary'] | df['had_close_general']

        try:
            notable = df[df['is_notable'] & (df['num_elections'] > 0)]
            if not notable.empty:
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                contested = [
                    notable[notable['ever_contested_any']]['familiarity_rate'],
                    notable[~notable['ever_contested_any']]['familiarity_rate']
                ]
                axes[0].boxplot(contested, labels=['Ever Contested', 'Never Contested'])
                axes[0].set_ylabel('Familiarity Rate (%)')
                axes[0].set_title('Contested Elections (Any)\nNotable Prosecutors', fontweight='bold')
                axes[0].grid(axis='y', alpha=0.3)

                close = [
                    notable[notable['had_close_any']]['familiarity_rate'],
                    notable[~notable['had_close_any']]['familiarity_rate']
                ]
                axes[1].boxplot(close, labels=['Had Close Race', 'No Close Race'])
                axes[1].set_ylabel('Familiarity Rate (%)')
                axes[1].set_title('Close Elections (Margin < 55%)\nNotable Prosecutors', fontweight='bold')
                axes[1].grid(axis='y', alpha=0.3)
                self._savefig(viz_dir / 'familiarity_elections_notable.png')
        except Exception as exc:
            print(f"[Viz] Notable election viz failed: {exc}")

        try:
            state = df[(~df['is_notable']) & (df['num_elections'] > 0)]
            if len(state) > 5:
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                contested = [
                    state[state['ever_contested_any']]['familiarity_rate'],
                    state[~state['ever_contested_any']]['familiarity_rate']
                ]
                axes[0].boxplot(contested, labels=['Ever Contested', 'Never Contested'])
                axes[0].set_ylabel('Familiarity Rate (%)')
                axes[0].set_title('Contested Elections (Any)\nState Prosecutors', fontweight='bold')
                axes[0].grid(axis='y', alpha=0.3)

                close = [
                    state[state['had_close_any']]['familiarity_rate'],
                    state[~state['had_close_any']]['familiarity_rate']
                ]
                axes[1].boxplot(close, labels=['Had Close Race', 'No Close Race'])
                axes[1].set_ylabel('Familiarity Rate (%)')
                axes[1].set_title('Close Elections (Margin < 55%)\nState Prosecutors', fontweight='bold')
                axes[1].grid(axis='y', alpha=0.3)
                self._savefig(viz_dir / 'familiarity_elections_state.png')
        except Exception as exc:
            print(f"[Viz] State election viz failed: {exc}")

        try:
            notable = df[df['is_notable'] & (df['num_elections'] > 0)]
            if not notable.empty:
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                # Primary margin scatter
                ax = axes[0]
                primary_mask = notable['closest_primary_margin'].notna()
                if primary_mask.sum() > 2:
                    ax.scatter(notable.loc[primary_mask, 'closest_primary_margin'],
                               notable.loc[primary_mask, 'familiarity_rate'],
                               alpha=0.6, s=80, color='steelblue')
                    x = notable.loc[primary_mask, 'closest_primary_margin']
                    y = notable.loc[primary_mask, 'familiarity_rate']
                    coef = np.polyfit(x, y, 1)
                    ax.plot(x, np.poly1d(coef)(x), 'r--', alpha=0.8, linewidth=2)
                ax.set_xlabel('Closest Primary Margin (%)')
                ax.set_ylabel('Familiarity Rate (%)')
                ax.set_title('Primary Election Competitiveness\nvs Familiarity', fontweight='bold')
                ax.grid(alpha=0.3)

                ax = axes[1]
                general_mask = notable['closest_general_margin'].notna()
                if general_mask.sum() > 2:
                    ax.scatter(notable.loc[general_mask, 'closest_general_margin'],
                               notable.loc[general_mask, 'familiarity_rate'],
                               alpha=0.6, s=80, color='darkgreen')
                    x = notable.loc[general_mask, 'closest_general_margin']
                    y = notable.loc[general_mask, 'familiarity_rate']
                    coef = np.polyfit(x, y, 1)
                    ax.plot(x, np.poly1d(coef)(x), 'r--', alpha=0.8, linewidth=2)
                ax.set_xlabel('Closest General Margin (%)')
                ax.set_ylabel('Familiarity Rate (%)')
                ax.set_title('General Election Competitiveness\nvs Familiarity', fontweight='bold')
                ax.grid(alpha=0.3)

                self._savefig(viz_dir / 'margin_familiarity_scatter.png')
        except Exception as exc:
            print(f"[Viz] Margin scatter viz failed: {exc}")

    def generate_tier1_visualization(self):
        df = getattr(self.analyzer, 'df_matched_all', pd.DataFrame()).copy()
        if df.empty:
            return

        viz_dir = Path(self.config.VIZ_DIR)
        try:
            notable = df[df['is_notable']]
            state = df[~df['is_notable']]

            challengers = notable[notable['ever_ran_as_challenger']]
            incumbents = notable[notable['ever_ran_as_incumbent']]
            state_challengers = state[state['ever_ran_as_challenger']]
            state_incumbents = state[state['ever_ran_as_incumbent']]

            notable_with_elections = notable[notable['num_elections'] > 0]
            state_with_elections = state[state['num_elections'] > 0]

            prog = df[df['is_progressive']]
            trad = df[df['is_traditional']]

            fig, axes = plt.subplots(2, 3, figsize=(18, 10))

            axes[0, 0].boxplot([challengers['familiarity_rate'], incumbents['familiarity_rate']],
                               labels=['Challenger', 'Incumbent'])
            axes[0, 0].set_ylabel('Familiarity Rate (%)')
            axes[0, 0].set_title('RQ3: Notable Prosecutors\nChallenger vs. Incumbent', fontweight='bold')
            axes[0, 0].grid(axis='y', alpha=0.3)

            axes[0, 1].boxplot([state_challengers['familiarity_rate'], state_incumbents['familiarity_rate']],
                               labels=['Challenger', 'Incumbent'])
            axes[0, 1].set_ylabel('Familiarity Rate (%)')
            axes[0, 1].set_title('RQ3: State Prosecutors\nChallenger vs. Incumbent', fontweight='bold')
            axes[0, 1].grid(axis='y', alpha=0.3)

            ax = axes[0, 2]
            ax.scatter(notable_with_elections['num_elections'], notable_with_elections['familiarity_rate'], alpha=0.6, s=80)
            if len(notable_with_elections) > 1:
                x = notable_with_elections['num_elections']
                y = notable_with_elections['familiarity_rate']
                coef = np.polyfit(x, y, 1)
                x_line = np.linspace(x.min(), x.max(), 100)
                ax.plot(x_line, np.poly1d(coef)(x_line), 'r--', alpha=0.8, linewidth=2)
            ax.set_xlabel('Number of Elections')
            ax.set_ylabel('Familiarity Rate (%)')
            ax.set_title('RQ4: Notable Prosecutors\nElections vs. Familiarity', fontweight='bold')
            ax.grid(alpha=0.3)

            ax = axes[1, 0]
            ax.scatter(state_with_elections['num_elections'], state_with_elections['familiarity_rate'], alpha=0.4, s=60)
            if len(state_with_elections) > 1:
                x = state_with_elections['num_elections']
                y = state_with_elections['familiarity_rate']
                coef = np.polyfit(x, y, 1)
                x_line = np.linspace(x.min(), x.max(), 100)
                ax.plot(x_line, np.poly1d(coef)(x_line), 'r--', alpha=0.8, linewidth=2)
            ax.set_xlabel('Number of Elections')
            ax.set_ylabel('Familiarity Rate (%)')
            ax.set_title('RQ4: State Prosecutors\nElections vs. Familiarity', fontweight='bold')
            ax.grid(alpha=0.3)

            prog_data = [
                prog[prog['ever_contested_primary']]['familiarity_rate'],
                prog[~prog['ever_contested_primary']]['familiarity_rate'],
                prog[prog['ever_contested_general']]['familiarity_rate'],
                prog[~prog['ever_contested_general']]['familiarity_rate']
            ]
            axes[1, 1].boxplot(prog_data, positions=[1, 2, 4, 5])
            axes[1, 1].set_xticks([1, 2, 4, 5])
            axes[1, 1].set_xticklabels(['Cont.\nPrim', 'Uncont.\nPrim', 'Cont.\nGen', 'Uncont.\nGen'])
            axes[1, 1].set_ylabel('Familiarity Rate (%)')
            axes[1, 1].set_title('RQ5: Progressive Prosecutors\nPrimary vs. General Effects', fontweight='bold')
            axes[1, 1].grid(axis='y', alpha=0.3)

            trad_data = [
                trad[trad['ever_contested_primary']]['familiarity_rate'],
                trad[~trad['ever_contested_primary']]['familiarity_rate'],
                trad[trad['ever_contested_general']]['familiarity_rate'],
                trad[~trad['ever_contested_general']]['familiarity_rate']
            ]
            axes[1, 2].boxplot(trad_data, positions=[1, 2, 4, 5])
            axes[1, 2].set_xticks([1, 2, 4, 5])
            axes[1, 2].set_xticklabels(['Cont.\nPrim', 'Uncont.\nPrim', 'Cont.\nGen', 'Uncont.\nGen'])
            axes[1, 2].set_ylabel('Familiarity Rate (%)')
            axes[1, 2].set_title('RQ5: Traditional Prosecutors\nPrimary vs. General Effects', fontweight='bold')
            axes[1, 2].grid(axis='y', alpha=0.3)

            plt.tight_layout()
            plt.savefig(viz_dir / 'tier1_research_questions.png', dpi=200, bbox_inches='tight')
            plt.close()
        except Exception as exc:
            print(f"[Viz] Tier1 visualization failed: {exc}")

    # ------------------------------------------------------------------
    # Legacy visualization helpers
    # ------------------------------------------------------------------

    def _viz_rankings_top20(self, notable_df, viz_dir):
        reliable = notable_df[notable_df['Substantive_Ratings'] >= 30].copy()
        if reliable.empty:
            return
        top20 = reliable.sort_values('Mean_Score', ascending=False).head(20)

        fig, ax = plt.subplots(figsize=(12, 10))
        y_pos = np.arange(len(top20))
        colors = plt.cm.RdYlGn((top20['Mean_Score'] - 1) / 3)
        ax.barh(y_pos, top20['Mean_Score'], color=colors, alpha=0.8)
        for i, (_, row) in enumerate(top20.iterrows()):
            ax.text(row['Mean_Score'] + 0.05, i, f"{row['Mean_Score']:.2f}", va='center', fontweight='bold', fontsize=9)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"{row['Name']} ({row['Location']})" for _, row in top20.iterrows()], fontsize=10)
        ax.set_xlabel('Progressiveness Score (1-4 Scale)', fontweight='bold', fontsize=11)
        ax.set_title('Top 20 Most Progressive District Attorneys', fontweight='bold', fontsize=14, pad=20)
        ax.set_xlim(1, 4)
        ax.axvline(x=2.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        self._savefig(viz_dir / 'viz_1_rankings.png')

    def _viz_all_notables(self, notable_df, viz_dir):
        reliable = notable_df[notable_df['Substantive_Ratings'] >= 10].copy()
        if reliable.empty:
            return
        reliable = reliable.sort_values('Mean_Score', ascending=False)

        fig, ax = plt.subplots(figsize=(16, 14))
        y_pos = np.arange(len(reliable))
        colors = plt.cm.RdYlGn((reliable['Mean_Score'] - 1) / 3)
        ax.barh(y_pos, reliable['Mean_Score'], color=colors, alpha=0.7)
        for i, (_, row) in enumerate(reliable.iterrows()):
            ax.text(row['Mean_Score'] + 0.05, i, f"{row['Mean_Score']:.2f} (n={int(row['Substantive_Ratings'])})",
                    va='center', fontsize=8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"{row['Name']} ({row['Location']})" for _, row in reliable.iterrows()], fontsize=9)
        ax.set_xlabel('Progressiveness Score (1-4 Scale)', fontweight='bold', fontsize=12)
        ax.set_title('Complete Rankings: 50 Notable District Attorneys', fontweight='bold', fontsize=16, pad=20)
        ax.set_xlim(1, 4)
        ax.axvline(x=2.5, color='gray', linestyle='--', alpha=0.5, linewidth=1, label='Midpoint (2.5)')
        ax.axvline(x=3, color='green', linestyle='--', alpha=0.5, linewidth=1, label='Progressive Threshold (≥3)')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        ax.legend(loc='lower right')
        self._savefig(viz_dir / 'viz_2_all_50_das.png')

    def _viz_position_effects(self, notable_df, viz_dir):
        reliable = notable_df[
            (notable_df['Substantive_Ratings'] >= 30) &
            notable_df['Academic_Mean'].notna() &
            notable_df['Defense_Mean'].notna() &
            notable_df['Prosecutor_Mean'].notna()
        ].copy()
        if reliable.empty:
            return
        top = reliable.sort_values('Substantive_Ratings', ascending=False).head(15)

        fig, ax = plt.subplots(figsize=(14, 10))
        x = np.arange(len(top))
        width = 0.25
        ax.bar(x - width, top['Academic_Mean'], width, label='Academic', color='#3498db', alpha=0.8)
        ax.bar(x, top['Defense_Mean'], width, label='Defense Attorney', color='#e74c3c', alpha=0.8)
        ax.bar(x + width, top['Prosecutor_Mean'], width, label='Prosecutor', color='#2ecc71', alpha=0.8)
        ax.set_ylabel('Mean Progressiveness Score', fontweight='bold', fontsize=11)
        ax.set_title('How Different Professionals Rate the Same Prosecutors', fontweight='bold', fontsize=14, pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(top['Name'], rotation=45, ha='right', fontsize=9)
        ax.legend(loc='upper right', fontsize=10)
        ax.set_ylim(1, 4)
        ax.axhline(y=2.5, color='gray', linestyle='--', alpha=0.3)
        ax.grid(axis='y', alpha=0.3)
        self._savefig(viz_dir / 'viz_3_position_effects.png')

    def _viz_familiarity_crisis(self, notable_df, viz_dir):
        reliable = notable_df[notable_df['Substantive_Ratings'] >= 10].copy()
        if reliable.empty:
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        top20 = reliable.nlargest(20, 'Familiarity_Rate')
        y_pos = np.arange(len(top20))
        ax1.barh(y_pos, top20['Familiarity_Rate'], color='#3498db', alpha=0.7)
        for i, (_, row) in enumerate(top20.iterrows()):
            ax1.text(row['Familiarity_Rate'] + 1, i, f"{row['Familiarity_Rate']:.0f}%", va='center', fontsize=8)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(top20['Name'], fontsize=9)
        ax1.set_xlabel('Familiarity Rate (%)', fontweight='bold')
        ax1.set_title('Top 20 Most Familiar Prosecutors', fontweight='bold', fontsize=12)
        ax1.invert_yaxis()
        ax1.grid(axis='x', alpha=0.3)

        scatter_df = reliable[reliable['Substantive_Ratings'] >= 30]
        scatter = ax2.scatter(scatter_df['Familiarity_Rate'], scatter_df['Mean_Score'],
                              c=scatter_df['Mean_Score'], cmap='RdYlGn', s=100, alpha=0.6, vmin=1, vmax=4)
        for _, row in scatter_df.head(10).iterrows():
            ax2.annotate(row['Name'], (row['Familiarity_Rate'], row['Mean_Score']),
                         xytext=(5, 5), textcoords='offset points', fontsize=7, alpha=0.7)
        ax2.set_xlabel('Familiarity Rate (%)', fontweight='bold')
        ax2.set_ylabel('Mean Progressiveness Score', fontweight='bold')
        ax2.set_title('Familiarity vs Progressiveness', fontweight='bold', fontsize=12)
        ax2.grid(alpha=0.3)
        ax2.set_ylim(1, 4)
        plt.colorbar(scatter, ax=ax2, label='Progressiveness')

        plt.tight_layout()
        plt.savefig(viz_dir / 'viz_4_familiarity.png', dpi=200, bbox_inches='tight')
        plt.close()

    def _viz_transitions(self, transitions_df, viz_dir):
        if transitions_df.empty:
            return
        fig, ax = plt.subplots(figsize=(14, 10))
        ordered = transitions_df.sort_values('Change')
        y_pos = np.arange(len(ordered))
        colors = ['#e74c3c' if x < 0 else '#2ecc71' for x in ordered['Change']]
        ax.barh(y_pos, ordered['Change'], color=colors, alpha=0.7)
        for i, (_, row) in enumerate(ordered.iterrows()):
            label = f"{row['Predecessor']} → {row['Successor']}\n{row['Jurisdiction']}"
            ax.text(-0.1, i, label, va='center', ha='right', fontsize=9)
            offset = 0.05 if row['Change'] > 0 else -0.05
            ha = 'left' if row['Change'] > 0 else 'right'
            ax.text(row['Change'] + offset, i, f"{row['Change']:+.2f}", va='center', ha=ha, fontweight='bold', fontsize=9)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([''] * len(ordered))
        ax.set_xlabel('Change in Progressiveness Score', fontweight='bold', fontsize=12)
        ax.set_title('Electoral Transitions: How Jurisdictions Shifted', fontweight='bold', fontsize=14, pad=20)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax.grid(axis='x', alpha=0.3)
        ax.set_xlim(-3, 1)
        traditional_patch = mpatches.Patch(color='#e74c3c', alpha=0.7, label='Shift toward Traditional')
        progressive_patch = mpatches.Patch(color='#2ecc71', alpha=0.7, label='Shift toward Progressive')
        ax.legend(handles=[traditional_patch, progressive_patch], loc='lower right')
        self._savefig(viz_dir / 'viz_5_transitions.png')

    def _viz_controversy(self, notable_df, viz_dir):
        reliable = notable_df[notable_df['Substantive_Ratings'] >= 30].copy()
        if reliable.empty:
            return
        most_controversial = reliable.nlargest(15, 'Std_Dev')
        fig, ax = plt.subplots(figsize=(14, 10))
        y_pos = np.arange(len(most_controversial))
        colors = plt.cm.RdYlGn((most_controversial['Mean_Score'] - 1) / 3)
        ax.barh(y_pos, most_controversial['Std_Dev'], color=colors, alpha=0.7)
        for i, (_, row) in enumerate(most_controversial.iterrows()):
            ax.text(row['Std_Dev'] + 0.02, i, f"σ={row['Std_Dev']:.2f} | μ={row['Mean_Score']:.2f}", va='center', fontsize=9)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"{row['Name']} ({row['Location']})" for _, row in most_controversial.iterrows()], fontsize=10)
        ax.set_xlabel('Standard Deviation (Disagreement)', fontweight='bold', fontsize=11)
        ax.set_title('Most Controversial Prosecutors: Highest Disagreement in Ratings', fontweight='bold', fontsize=14, pad=20)
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        ax.text(0.95, 0.05,
                'Bar color indicates mean score\nRed = Traditional, Yellow = Moderate, Green = Progressive',
                transform=ax.transAxes, fontsize=9, va='bottom', ha='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        self._savefig(viz_dir / 'viz_6_controversy.png')

    def _viz_geographic(self, notable_df, viz_dir):
        state_counts = {}
        for _, row in notable_df.iterrows():
            location = row['Location']
            if isinstance(location, str) and ', ' in location:
                state = location.split(', ')[-1]
                bucket = state_counts.setdefault(state, {'count': 0, 'total_score': 0.0, 'scores': []})
                bucket['count'] += 1
                if not pd.isna(row['Mean_Score']):
                    bucket['total_score'] += row['Mean_Score']
                    bucket['scores'].append(row['Mean_Score'])

        summary = []
        for state, data in state_counts.items():
            avg = np.nan
            if data['count'] > 0 and data['scores']:
                avg = np.mean(data['scores'])
            summary.append({'state': state, 'n_das': data['count'], 'avg_score': avg})

        state_df = pd.DataFrame(summary).sort_values('n_das', ascending=False).head(15)
        if state_df.empty:
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        y_pos = np.arange(len(state_df))
        ax1.barh(y_pos, state_df['n_das'], color='#3498db', alpha=0.7)
        for i, (_, row) in enumerate(state_df.iterrows()):
            ax1.text(row['n_das'] + 0.1, i, f"{row['n_das']}", va='center', fontsize=10)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(state_df['state'], fontsize=11)
        ax1.set_xlabel('Number of Prosecutors in National Sample', fontweight='bold')
        ax1.set_title('States with Most Prosecutors in Study', fontweight='bold', fontsize=12)
        ax1.invert_yaxis()
        ax1.grid(axis='x', alpha=0.3)

        scored = state_df[state_df['avg_score'].notna()].sort_values('avg_score', ascending=False)
        y_pos2 = np.arange(len(scored))
        ax2.barh(y_pos2, scored['avg_score'], color=plt.cm.RdYlGn((scored['avg_score'] - 1) / 3), alpha=0.7)
        for i, (_, row) in enumerate(scored.iterrows()):
            ax2.text(row['avg_score'] + 0.02, i, f"{row['avg_score']:.2f}", va='center', fontsize=9)
        ax2.set_yticks(y_pos2)
        ax2.set_yticklabels(scored['state'], fontsize=11)
        ax2.set_xlabel('Average Progressiveness Score', fontweight='bold')
        ax2.set_title('Average Progressiveness by State', fontweight='bold', fontsize=12)
        ax2.invert_yaxis()
        ax2.grid(axis='x', alpha=0.3)
        self._savefig(viz_dir / 'viz_7_geographic.png')

    def _viz_national_vs_state(self, notable_df, state_df, viz_dir):
        reliable_national = notable_df[notable_df['Substantive_Ratings'] >= Config.MIN_RATINGS_THRESHOLD]
        if reliable_national.empty or state_df.empty:
            return

        national_mean = reliable_national['Mean_Score'].mean()
        state_mean = state_df['mean_score'].mean()

        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.hist(reliable_national['Mean_Score'], bins=20, alpha=0.6,
                 color='#e74c3c', label=f'National DAs (μ={national_mean:.2f})', edgecolor='black')
        state_scores = []
        for _, row in state_df.iterrows():
            weight = max(int(row['total_ratings'] / 10), 1)
            state_scores.extend([row['mean_score']] * weight)
        if state_scores:
            ax1.hist(state_scores, bins=20, alpha=0.6,
                     color='#3498db', label=f'State DAs (μ={state_mean:.2f})', edgecolor='black')
        ax1.set_xlabel('Progressiveness Score', fontweight='bold')
        ax1.set_ylabel('Frequency', fontweight='bold')
        ax1.set_title('Score Distribution: National vs State', fontweight='bold', fontsize=13)
        ax1.legend()
        ax1.grid(alpha=0.3)
        ax1.set_xlim(1, 4)

        ax2 = fig.add_subplot(gs[0, 1])
        nat_prog = len(reliable_national[reliable_national['Mean_Score'] >= 3])
        nat_trad = len(reliable_national[reliable_national['Mean_Score'] <= 2])
        nat_mod = len(reliable_national) - nat_prog - nat_trad
        state_prog = len(state_df[state_df['mean_score'] >= 3])
        state_trad = len(state_df[state_df['mean_score'] <= 2])
        state_mod = len(state_df) - state_prog - state_trad
        x = np.arange(3)
        width = 0.35
        nat_counts = np.array([nat_trad, nat_mod, nat_prog], dtype=float)
        state_counts = np.array([state_trad, state_mod, state_prog], dtype=float)
        nat_pct = (nat_counts / len(reliable_national) * 100) if len(reliable_national) else np.zeros(3)
        state_pct = (state_counts / len(state_df) * 100) if len(state_df) else np.zeros(3)
        ax2.bar(x - width / 2, nat_pct, width, label='National DAs', color='#e74c3c', alpha=0.7)
        ax2.bar(x + width / 2, state_pct, width, label='State DAs', color='#3498db', alpha=0.7)
        ax2.set_ylabel('Percentage', fontweight='bold')
        ax2.set_title('Ideological Distribution', fontweight='bold', fontsize=13)
        ax2.set_xticks(x)
        ax2.set_xticklabels(['Traditional\n(≤2)', 'Moderate\n(>2,<3)', 'Progressive\n(≥3)'])
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)

        ax3 = fig.add_subplot(gs[1, 0])
        top10 = reliable_national.nlargest(10, 'Mean_Score')
        y_pos = np.arange(len(top10))
        ax3.barh(y_pos, top10['Mean_Score'], color=plt.cm.RdYlGn((top10['Mean_Score'] - 1) / 3), alpha=0.7)
        for i, (_, row) in enumerate(top10.iterrows()):
            ax3.text(row['Mean_Score'] + 0.05, i, f"{row['Mean_Score']:.2f}", va='center', fontsize=8)
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(top10['Name'], fontsize=9)
        ax3.set_xlabel('Progressiveness Score', fontweight='bold')
        ax3.set_title('Top 10 National DAs', fontweight='bold', fontsize=13)
        ax3.invert_yaxis()
        ax3.grid(axis='x', alpha=0.3)
        ax3.set_xlim(1, 4)

        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis('off')
        gap = national_mean - state_mean
        gap_pct = (gap / 3) * 100
        most_known = reliable_national.nlargest(1, 'Familiarity_Rate')
        if not most_known.empty:
            most_known_name = most_known['Name'].iloc[0]
            most_known_pct = most_known['Familiarity_Rate'].iloc[0]
        else:
            most_known_name = 'N/A'
            most_known_pct = np.nan
        avg_unfam = 100 - reliable_national['Familiarity_Rate'].mean()
        summary_text = f"""
**KEY FINDINGS & OBSERVATIONS:**

**1. National vs. State Gap**
   • **National DAs:** Rated at **{national_mean:.2f}**
   • **State-Level DAs:** Rated at **{state_mean:.2f}**
   • **Gap:** **{gap:.2f}** points (~{gap_pct:.0f}% of scale)

**2. Familiarity Concentration**
   • Average unfamiliarity with national DAs: {avg_unfam:.0f}%
   • Most known: {most_known_name} ({most_known_pct:.0f}%)
"""
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
                 fontsize=11, verticalalignment='top',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0f0f0', alpha=0.8))

        plt.suptitle('The Two-Tier System of Prosecutor Perception: National vs. State',
                     fontsize=16, fontweight='bold', y=0.98)
        plt.savefig(viz_dir / 'viz_8_national_vs_state.png', dpi=200, bbox_inches='tight')
        plt.close()

    def _familiarity_breakdown(self):
        df = self.analyzer.df_survey.copy()
        notable_cols = self.analyzer.notable_cols
        breakdown = {}
        if not notable_cols:
            return breakdown

        rated_mask = df[notable_cols].notna().any(axis=1)
        respondents_who_rated = rated_mask.sum()
        total_opportunities = respondents_who_rated * len(notable_cols)

        explicit_nf = 0
        left_blank = 0
        substantive = 0

        for _, row in df[rated_mask][notable_cols].iterrows():
            explicit_nf += (row == 'Not Familiar').sum()
            left_blank += row.isna().sum()
            substantive += len(row.dropna()) - (row == 'Not Familiar').sum()

        breakdown['notable'] = {
            'respondents_who_rated': int(respondents_who_rated),
            'total_opportunities': int(total_opportunities),
            'explicit_not_familiar': int(explicit_nf),
            'left_blank': int(left_blank),
            'substantive': int(substantive)
        }

        state_cols = self.analyzer.state_da_cols
        if state_cols:
            if 'more' in df.columns:
                state_gate = df['more'] == 'Show me the prosecutors from my state'
            else:
                state_gate = pd.Series(True, index=df.index)

            state_to_cols = {}
            for col in state_cols:
                state_name = col.rsplit('_', 1)[0]
                state_to_cols.setdefault(state_name, []).append(col)

            total_opp = explicit_nf_state = left_blank_state = substantive_state = 0
            engaged_state_counts = 0

            for _, row in df[state_gate].iterrows():
                respondent_state = row.get('RespondentState') or row.get('Q1')
                if pd.isna(respondent_state):
                    continue
                state_specific_cols = state_to_cols.get(respondent_state, [])
                if not state_specific_cols:
                    continue
                engaged = False
                for col in state_specific_cols:
                    val = row[col]
                    total_opp += 1
                    if pd.isna(val):
                        left_blank_state += 1
                    elif val == 'Not Familiar':
                        engaged = True
                        explicit_nf_state += 1
                    else:
                        engaged = True
                        substantive_state += 1
                if engaged:
                    engaged_state_counts += 1

            breakdown['state'] = {
                'respondents_who_rated': int(engaged_state_counts),
                'total_opportunities': int(total_opp),
                'explicit_not_familiar': int(explicit_nf_state),
                'left_blank': int(left_blank_state),
                'substantive': int(substantive_state)
            }

        return breakdown

    def produce_extended_summary(self):
        print_section("WRITING INTEGRATED SUMMARY")
        breakdown = self._familiarity_breakdown()
        df = self.analyzer.df_prosecutors
        dfm = getattr(self.analyzer, 'df_matched_all', pd.DataFrame())

        lines = []
        lines.append("=" * 90)
        lines.append("INTEGRATED MASTER SUMMARY")
        lines.append("=" * 90)
        lines.append("")
        lines.append(f"Total respondents (after consent): {self.analyzer.total_respondents}")
        lines.append(f"Completed national section: {self.analyzer.completed_national_section}")
        lines.append(f"Total prosecutors analyzed: {len(df)} (Notable: {int(df['is_notable'].sum())}, State: {int((~df['is_notable']).sum())})")
        lines.append("")

        if breakdown.get('notable'):
            nb = breakdown['notable']
            lines.append("National (50 notable prosecutors):")
            lines.append(f"  Respondents who rated ≥1: {nb['respondents_who_rated']}")
            lines.append(f"  Opportunities: {nb['total_opportunities']:,}")
            if nb['total_opportunities']:
                lines.append(f"  Explicit 'Not Familiar': {nb['explicit_not_familiar']:,} ({nb['explicit_not_familiar']/nb['total_opportunities']*100:.1f}%)")
                lines.append(f"  Left blank: {nb['left_blank']:,} ({nb['left_blank']/nb['total_opportunities']*100:.1f}%)")
                lines.append(f"  Substantive ratings: {nb['substantive']:,} ({nb['substantive']/nb['total_opportunities']*100:.1f}%)")
            lines.append("")

        if breakdown.get('state'):
            sb = breakdown['state']
            lines.append("State prosecutor familiarity (state-engaged respondents):")
            lines.append(f"  Respondents providing ≥1 state rating: {sb['respondents_who_rated']}")
            lines.append(f"  Opportunities: {sb['total_opportunities']:,}")
            if sb['total_opportunities']:
                lines.append(f"  Explicit 'Not Familiar': {sb['explicit_not_familiar']:,} ({sb['explicit_not_familiar']/sb['total_opportunities']*100:.1f}%)")
                lines.append(f"  Left blank: {sb['left_blank']:,} ({sb['left_blank']/sb['total_opportunities']*100:.1f}%)")
                lines.append(f"  Substantive ratings: {sb['substantive']:,} ({sb['substantive']/sb['total_opportunities']*100:.1f}%)")
            lines.append("")

        notable = df[df['is_notable']]
        state = df[~df['is_notable']]
        lines.append(f"Mean familiarity — Notables: {notable['familiarity_rate'].mean():.2f}% | State: {state['familiarity_rate'].mean():.2f}%")
        lines.append(f"Median ideology score (filtered ≥{self.config.MIN_RATINGS_THRESHOLD} ratings): {self.analyzer.df_prosecutors_filtered_ideology['mean_score'].median():.2f}")
        lines.append("")

        filtered = self.analyzer.df_prosecutors_filtered_ideology
        if len(filtered):
            top10 = filtered[filtered['is_notable']].sort_values('mean_score', ascending=False).head(10)
            bot10 = filtered[filtered['is_notable']].sort_values('mean_score', ascending=True).head(10)
            lines.append("Top 10 most progressive notable prosecutors (by mean score):")
            for _, row in top10.iterrows():
                lines.append(f"  {row['name']:<35s}  Mean={row['mean_score']:.2f}  Familiarity={row['familiarity_rate']:.1f}%")
            lines.append("")
            lines.append("Top 10 most traditional notable prosecutors (by mean score):")
            for _, row in bot10.iterrows():
                lines.append(f"  {row['name']:<35s}  Mean={row['mean_score']:.2f}  Familiarity={row['familiarity_rate']:.1f}%")
            lines.append("")

        if len(dfm):
            lines.append("Election & recall context (matched sample):")
            lines.append(f"  Prosecutors matched with elections: {len(dfm)}")
            lines.append(f"  Ever contested any election: {dfm['ever_contested_any'].mean()*100:.1f}%")
            lines.append(f"  Close general elections (≤{self.config.CLOSE_MARGIN_THRESHOLD}% margin): {dfm['had_close_general'].mean()*100:.1f}%")
            lines.append(f"  Close primary elections (≤{self.config.CLOSE_MARGIN_THRESHOLD}% margin): {dfm['had_close_primary'].mean()*100:.1f}%")
            recalled = dfm[dfm['recalled'] == True]
            if len(recalled):
                lines.append("  Recalled prosecutors:")
                for _, row in recalled.iterrows():
                    lines.append(f"    {row['name']}: closest general margin={row['closest_general_margin'] if pd.notna(row['closest_general_margin']) else 'N/A'} | familiarity={row['familiarity_rate']:.1f}%")
            lines.append("")

        summary_path = Path(self.config.OUTPUT_DIR) / 'INTEGRATED_MASTER_SUMMARY.txt'
        summary_path.write_text("\n".join(lines), encoding='utf-8')
        print(f"✓ Integrated summary saved -> {summary_path}")
        return summary_path

    def produce_visibility_competition_report(self):
        """Write the extended visibility & competition analyses to disk."""

        results = self.analyzer.results.get('visibility_competition')
        if not results:
            print("⚠ Visibility/competition results not available.")
            return None

        def fmt_pct(value):
            return f"{value*100:.1f}%" if value is not None and pd.notna(value) else "N/A"

        def fmt_num(value, digits=2):
            return f"{value:.{digits}f}" if value is not None and pd.notna(value) else "N/A"

        lines = []
        lines.append("=" * 100)
        lines.append("EXTENDED VISIBILITY, COMPETITION, AND IDEOLOGY TESTS")
        lines.append("=" * 100)
        lines.append("")

        bivariate = results.get('bivariate_pattern') or {}
        lines.append("Bivariate Pattern: Familiarity and Progressiveness")
        lines.append("-" * 100)
        lines.append(
            f"Prosecutors with familiarity above 30% (n={bivariate.get('high_count', 0)}): mean progressiveness {fmt_num(bivariate.get('high_mean'))}"
        )
        lines.append(
            f"Prosecutors with familiarity below 10% (n={bivariate.get('low_count', 0)}): mean progressiveness {fmt_num(bivariate.get('low_mean'))}"
        )
        lines.append(
            f"Difference: {fmt_num(bivariate.get('difference'))} points on the 4-point scale"
        )
        if pd.notna(bivariate.get('t_stat')):
            lines.append(
                f"Welch’s t ≈ {fmt_num(bivariate.get('t_stat'))}; p = {fmt_num(bivariate.get('p_val'))}"
            )
        lines.append("")

        corr = results.get('familiarity_progressiveness_corr') or {}
        lines.append("Overall Familiarity-Progressiveness Correlation")
        lines.append("-" * 100)
        if corr:
            lines.append(
                f"Sample size: {corr.get('sample_size', 0)} prosecutors with ideology & familiarity"
            )
            lines.append(
                f"Pearson r = {fmt_num(corr.get('pearson_r'))}; p = {fmt_num(corr.get('pearson_p'))}"
            )
        else:
            lines.append("Insufficient data for correlation estimate.")
        lines.append("")

        national = results.get('national_contestation') or {}
        lines.append("Electoral Competition and Prosecutor Visibility")
        lines.append("-" * 100)
        lines.append("The merger of survey responses with comprehensive electoral data (10,328 election records) revealed ...")
        lines.append("")
        lines.append("The National Pattern: Competition and Visibility")
        lines.append("~" * 100)
        lines.append(
            f"Contested General Elections: mean familiarity {fmt_num(national.get('contested_mean'))}% — Prosecutors: {national.get('contested_count', 0)} (of {national.get('total_with_data', 0)} with election data)"
        )
        lines.append(
            f"Uncontested General Elections: mean familiarity {fmt_num(national.get('uncontested_mean'))}% — Prosecutors: {national.get('uncontested_count', 0)} (of {national.get('total_with_data', 0)})"
        )
        lines.append(
            f"Difference: {fmt_num(national.get('difference'))} percentage points; Welch’s t ≈ {fmt_num(national.get('t_stat'))} (p = {fmt_num(national.get('p_val'))})"
        )
        nat_diff = national.get('difference')
        if nat_diff is None or pd.isna(nat_diff):
            nat_interp = "Insufficient data to compare contested versus uncontested general elections."
        elif nat_diff > 0:
            nat_interp = "Contested general elections correspond with higher familiarity among notable prosecutors."
        elif nat_diff < 0:
            nat_interp = "Contested general elections correspond with lower familiarity among notable prosecutors."
        else:
            nat_interp = "Contested status shows no average familiarity difference among notable prosecutors."
        lines.append(f"Interpretation: {nat_interp}")
        lines.append("")

        state = results.get('state_contestation') or {}
        lines.append("The State Pattern: Contested Races and Familiarity")
        lines.append("~" * 100)
        lines.append(
            f"Uncontested General Elections: mean familiarity {fmt_num(state.get('uncontested_mean'))}% — Prosecutors: {state.get('uncontested_count', 0)} (of {state.get('total_with_data', 0)} with election data)"
        )
        lines.append(
            f"Contested General Elections: mean familiarity {fmt_num(state.get('contested_mean'))}% — Prosecutors: {state.get('contested_count', 0)} (of {state.get('total_with_data', 0)})"
        )
        lines.append(
            f"Difference: {fmt_num(state.get('difference'))} percentage points; Welch’s t ≈ {fmt_num(state.get('t_stat'))} (p = {fmt_num(state.get('p_val'))})"
        )
        lines.append("")
        lines.append("Primary Competition Effect:")
        lines.append(
            f"Close primary: mean familiarity {fmt_num(state.get('close_primary_mean'))}% (n = {state.get('close_primary_count', 0)})"
        )
        lines.append(
            f"Not close: mean familiarity {fmt_num(state.get('not_close_primary_mean'))}% (n = {state.get('not_close_primary_count', 0)})"
        )
        lines.append(
            f"Difference (close − not close): {fmt_num(state.get('close_primary_diff'))} pp; Welch’s t ≈ {fmt_num(state.get('close_primary_t'))} (p = {fmt_num(state.get('close_primary_p'))})"
        )
        lines.append(
            f"Correlation between primary margin and familiarity among state-level prosecutors (where a primary margin exists): r ≈ {fmt_num(state.get('primary_margin_corr'))} (p = {fmt_num(state.get('primary_margin_p'))})"
        )
        state_diff = state.get('difference')
        if state_diff is None or pd.isna(state_diff):
            state_interp = "Insufficient data to compare contested versus uncontested general elections for state prosecutors."
        elif state_diff > 0:
            state_interp = "Contested general elections correspond with higher familiarity among state-level prosecutors."
        elif state_diff < 0:
            state_interp = "State-level familiarity is slightly lower when prosecutors face contested general elections."
        else:
            state_interp = "General election contestation shows no meaningful familiarity difference among state prosecutors."
        lines.append(f"Interpretation: {state_interp}")
        lines.append("")

        overall = self.analyzer.results.get('contestation_analysis') or {}
        if overall:
            lines.append("Overall Contestation Effects (All Matched Prosecutors)")
            lines.append("-" * 100)
            lines.append(
                f"Ever contested any election (n={overall.get('contested_count', 0)}): mean familiarity {fmt_num(overall.get('contested_mean'))}%"
            )
            lines.append(
                f"Never contested (n={overall.get('uncontested_count', 0)}): mean familiarity {fmt_num(overall.get('uncontested_mean'))}%"
            )
            lines.append(
                f"Difference: {fmt_num(overall.get('difference'))} pp; t ≈ {fmt_num(overall.get('t_stat'))} (p = {fmt_num(overall.get('p_val'))})"
            )
            lines.append("")

        logistic = results.get('logistic_close_victory') or {}
        lines.append("Predictive Model: What Predicts Close Victory?")
        lines.append("-" * 100)
        total = logistic.get('sample_size', 0)
        close = logistic.get('close_victories', 0)
        comfortable = logistic.get('comfortable_victories', 0)
        lines.append(
            f"Among {total} prosecutors with general election data:"
        )
        lines.append(
            f"Close victories: {close} ({fmt_pct(close / total if total else np.nan)})"
        )
        lines.append(
            f"Comfortable victories: {comfortable} ({fmt_pct(comfortable / total if total else np.nan)})"
        )
        model = logistic.get('model')
        if model is not None:
            for param in ['ever_ran_as_challenger', 'mean_score', 'ever_contested_primary']:
                if param in model.params:
                    lines.append(
                        f"{param.replace('_', ' ').title()}: β = {fmt_num(model.params[param])}; p = {fmt_num(model.pvalues[param])}"
                    )
        lines.append("")

        pathway = results.get('vulnerability_pathway') or {}
        lines.append("The Pathway to Vulnerability")
        lines.append("-" * 100)
        lines.append(
            f"Progressive candidates enter as challengers {fmt_pct(pathway.get('progressive_challenger_rate'))} vs. {fmt_pct(pathway.get('non_progressive_challenger_rate'))} for non-progressive"
        )
        lines.append(
            f"Challengers win with narrower margins {fmt_num(pathway.get('challenger_margin_difference'))}"
        )
        if pathway.get('challenger_model') is not None and 'mean_score' in pathway['challenger_model'].params:
            lines.append(
                f"Logistic regression: β = {fmt_num(pathway['challenger_model'].params['mean_score'])}"
            )
        lines.append("")

        ideology = results.get('contestation_by_ideology') or {}
        lines.append("Contestation Rates by Ideology")
        lines.append("-" * 100)
        counts = ideology.get('counts', {})
        lines.append(
            f"Progressive: n={counts.get('Progressive', 0)} | Moderate: n={counts.get('Moderate', 0)} | Traditional: n={counts.get('Traditional', 0)}"
        )
        primary_rates = ideology.get('primary_rates', {})
        general_rates = ideology.get('general_rates', {})
        lines.append(
            f"Primary contestation — Progressive: {fmt_pct(primary_rates.get('Progressive'))}; Traditional: {fmt_pct(primary_rates.get('Traditional'))}"
        )
        if ideology.get('chi_primary'):
            if primary_rates.get('Progressive') is not None and primary_rates.get('Traditional') is not None:
                primary_diff = primary_rates.get('Progressive') - primary_rates.get('Traditional')
            else:
                primary_diff = None
            lines.append(
                f"Difference: {fmt_pct(primary_diff)} (χ² = {fmt_num(ideology['chi_primary'][0])}, p = {fmt_num(ideology['chi_primary'][1])})"
            )
        lines.append(
            f"General Election contestation — Progressive: {fmt_pct(general_rates.get('Progressive'))}; Traditional: {fmt_pct(general_rates.get('Traditional'))}"
        )
        if ideology.get('chi_general'):
            if general_rates.get('Progressive') is not None and general_rates.get('Traditional') is not None:
                general_diff = general_rates.get('Progressive') - general_rates.get('Traditional')
            else:
                general_diff = None
            lines.append(
                f"Difference: {fmt_pct(general_diff)} (χ² = {fmt_num(ideology['chi_general'][0])}, p = {fmt_num(ideology['chi_general'][1])})"
            )
        lines.append("")

        rater = results.get('rater_position_scores') or {}
        lines.append("Mean Ideology Ratings by Rater Position")
        lines.append("-" * 100)
        means = rater.get('means', {})
        lines.append(f"Prosecutors rating prosecutors: {fmt_num(means.get('Prosecutors rating prosecutors'))}")
        lines.append(f"Academics rating prosecutors: {fmt_num(means.get('Academics rating prosecutors'))}")
        lines.append(f"Defense attorneys rating prosecutors: {fmt_num(means.get('Defense attorneys rating prosecutors'))}")
        lines.append(f"Other professionals rating prosecutors: {fmt_num(means.get('Other professionals'))}")
        if rater:
            lines.append(
                f"Statistical Significance: F({fmt_num(rater.get('anova_df1'), 0)}, {fmt_num(rater.get('anova_df2'), 0)}) = {fmt_num(rater.get('anova_f'))}, p = {fmt_num(rater.get('anova_p'))}"
            )
        lines.append("")

        regional = results.get('regional_patterns') or {}
        lines.append("Regional Patterns")
        lines.append("-" * 100)
        for label, mean in (regional.get('means') or {}).items():
            lines.append(f"{label}: {fmt_num(mean)}")
        if regional:
            lines.append(
                f"Statistical Significance: F({fmt_num(regional.get('anova_df1'), 0)}, {fmt_num(regional.get('anova_df2'), 0)}) = {fmt_num(regional.get('anova_f'))}, p = {fmt_num(regional.get('anova_p'))}"
            )
        lines.append("")

        urban = results.get('urban_rural') or {}
        lines.append("Urban vs. Rural")
        lines.append("-" * 100)
        for label, mean in (urban.get('means') or {}).items():
            lines.append(f"{label}: mean {fmt_num(mean)}")
        if urban:
            lines.append(
                f"Correlation: County population and progressiveness r = {fmt_num(urban.get('population_correlation'))}"
            )
        lines.append("")

        geo = results.get('geographic_concentration') or {}
        lines.append("Geographic Concentration of Progressive Prosecution")
        lines.append("-" * 100)
        lines.append(
            f"Progressive prosecutors (score ≥2.5) concentrated in: California {geo.get('progressive_top10_ca', 0)} of top 10 most progressive; Major cities {geo.get('progressive_top10_major_city', 0)}; Blue states {geo.get('progressive_biden_states', 0)} of {geo.get('progressive_total', 0)}"
        )
        lines.append(
            f"Traditional prosecutors (score <2.5) concentrated in: Southern states {geo.get('traditional_southern_states', 0)}; Red states {geo.get('traditional_trump_states', 0)} of {geo.get('traditional_total', 0)}; Suburban/exurban county counts limited in source data"
        )
        lines.append("")

        report_path = Path(self.config.OUTPUT_DIR) / 'EXTENDED_VISIBILITY_COMPETITION.txt'
        report_path.write_text("\n".join(lines), encoding='utf-8')
        print(f"✓ Extended visibility & competition report saved -> {report_path}")
        return report_path

    def produce_unified_master_report(self, summary_path=None, visibility_path=None, rq_path=None):
        """Create a single consolidated text report for the narrative write-up."""

        print_section("WRITING MASTER REPORT DIGEST")

        def fmt_num(value, digits=2):
            return f"{value:.{digits}f}" if value is not None and pd.notna(value) else "N/A"

        def fmt_pct(value, digits=1):
            return f"{value:.{digits}f}%" if value is not None and pd.notna(value) else "N/A"

        def fmt_ratio(value, digits=1):
            return f"{value*100:.{digits}f}%" if value is not None and pd.notna(value) else "N/A"

        def add_section(title, width=90):
            lines.append(title)
            lines.append("-" * min(len(title), width))

        analyzer = self.analyzer
        results = analyzer.results

        lines = []
        lines.append("=" * 100)
        lines.append("MASTER FINDINGS REPORT DIGEST")
        lines.append("=" * 100)
        lines.append("")

        add_section("Data Overview")
        lines.append(f"Total respondents after consent: {analyzer.total_respondents}")
        lines.append(f"Completed national section: {analyzer.completed_national_section}")
        lines.append(f"Total prosecutors analyzed: {len(analyzer.df_prosecutors)}")
        state_counts = results.get('state_prosecutor_counts', {}) or {}
        if state_counts:
            top_states = sorted(state_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            state_text = ", ".join([f"{state}: {count}" for state, count in top_states])
            lines.append(f"Top states by number of rated prosecutors: {state_text}")
        position_counts = results.get('position_breakdown', {}) or {}
        if position_counts:
            lines.append("Respondent professional composition:")
            for label, count in sorted(position_counts.items(), key=lambda x: x[1], reverse=True):
                lines.append(f"  {label}: {count}")
        lines.append("")

        matching = results.get('matching_summary', {}) or {}
        add_section("Electoral Data Integration")
        if matching:
            lines.append(f"Notable prosecutors matched: {matching.get('matched_notable', 0)} of {matching.get('total_notable', 0)} ({fmt_ratio(matching.get('match_rate_notable'))})")
            lines.append(f"State prosecutors matched: {matching.get('matched_state', 0)} of {matching.get('total_state', 0)} ({fmt_ratio(matching.get('match_rate_state'))})")
            if matching.get('unmatched_notables'):
                lines.append("Unmatched notable prosecutors: " + ", ".join(matching['unmatched_notables']))
            if matching.get('unmatched_states'):
                lines.append("Unmatched state prosecutors (sample): " + ", ".join(matching['unmatched_states'][:10]))
        else:
            lines.append("Electoral matching summary unavailable.")
        lines.append("")

        fam = results.get('familiarity_comparison', {}) or {}
        add_section("Familiarity Patterns")
        if fam:
            lines.append(f"Mean familiarity — Notables: {fmt_num(fam.get('notable_mean'))}% | State: {fmt_num(fam.get('state_mean'))}%")
            lines.append(f"Difference (Notable − State): {fmt_num(fam.get('difference'))} pp (t = {fmt_num(fam.get('t_stat'))}, p = {fmt_num(fam.get('p_val'))})")
            top_df = fam.get('top_10')
            if isinstance(top_df, pd.DataFrame) and not top_df.empty:
                lines.append("Top 5 prosecutors by familiarity rate:")
                top5 = top_df.sort_values('familiarity_rate', ascending=False).head(5)
                for _, row in top5.iterrows():
                    lines.append(f"  {row['name']} — Familiarity {row['familiarity_rate']:.1f}% | Mean ideology {row['mean_score']:.2f}")
        else:
            lines.append("Familiarity comparison unavailable.")
        lines.append("")

        vis_results = results.get('visibility_competition', {}) or {}
        add_section("Familiarity, Ideology, and Competition")
        bivariate = vis_results.get('bivariate_pattern') or {}
        if bivariate:
            lines.append(f"Familiarity >30%: mean ideology {fmt_num(bivariate.get('high_mean'))} (n={bivariate.get('high_count', 0)})")
            lines.append(f"Familiarity <10%: mean ideology {fmt_num(bivariate.get('low_mean'))} (n={bivariate.get('low_count', 0)})")
            lines.append(f"Difference: {fmt_num(bivariate.get('difference'))} points (Welch’s t = {fmt_num(bivariate.get('t_stat'))}, p = {fmt_num(bivariate.get('p_val'))})")
        corr = vis_results.get('familiarity_progressiveness_corr') or {}
        if corr:
            lines.append(f"Pearson correlation (familiarity vs ideology): r = {fmt_num(corr.get('pearson_r'))} (p = {fmt_num(corr.get('pearson_p'))}, n = {corr.get('sample_size', 0)})")
        national = vis_results.get('national_contestation') or {}
        if national:
            lines.append(f"Notable contested general elections: familiarity {fmt_num(national.get('contested_mean'))}% (n={national.get('contested_count', 0)})")
            lines.append(f"Notable uncontested general elections: familiarity {fmt_num(national.get('uncontested_mean'))}% (n={national.get('uncontested_count', 0)})")
            lines.append(f"Difference: {fmt_num(national.get('difference'))} pp (t = {fmt_num(national.get('t_stat'))}, p = {fmt_num(national.get('p_val'))})")
        state = vis_results.get('state_contestation') or {}
        if state:
            lines.append(f"State contested general elections: familiarity {fmt_num(state.get('contested_mean'))}% (n={state.get('contested_count', 0)})")
            lines.append(f"State uncontested general elections: familiarity {fmt_num(state.get('uncontested_mean'))}% (n={state.get('uncontested_count', 0)})")
            lines.append(f"Difference: {fmt_num(state.get('difference'))} pp (t = {fmt_num(state.get('t_stat'))}, p = {fmt_num(state.get('p_val'))})")
            lines.append(f"Close primary vs not close: {fmt_num(state.get('close_primary_mean'))}% vs {fmt_num(state.get('not_close_primary_mean'))}% (Δ = {fmt_num(state.get('close_primary_diff'))} pp)")
            lines.append(f"Primary margin correlation with familiarity: r = {fmt_num(state.get('primary_margin_corr'))} (p = {fmt_num(state.get('primary_margin_p'))})")
        overall_contestation = results.get('contestation_analysis') or {}
        if overall_contestation:
            lines.append(f"Ever contested any election: familiarity {fmt_num(overall_contestation.get('contested_mean'))}% (n={overall_contestation.get('contested_count', 0)})")
            lines.append(f"Never contested: familiarity {fmt_num(overall_contestation.get('uncontested_mean'))}% (n={overall_contestation.get('uncontested_count', 0)})")
            lines.append(f"Difference: {fmt_num(overall_contestation.get('difference'))} pp (t = {fmt_num(overall_contestation.get('t_stat'))}, p = {fmt_num(overall_contestation.get('p_val'))})")
        lines.append("")

        add_section("Incumbency and Challenger Pathways")
        incumbency = results.get('incumbency_analysis') or {}
        if incumbency:
            lines.append(f"Incumbents (n={incumbency.get('incumbent_count', 0)}): familiarity {fmt_num(incumbency.get('incumbent_mean'))}%")
            lines.append(f"Challengers (n={incumbency.get('challenger_count', 0)}): familiarity {fmt_num(incumbency.get('challenger_mean'))}%")
            lines.append(f"Difference: {fmt_num(incumbency.get('difference'))} pp (t = {fmt_num(incumbency.get('t_stat'))}, p = {fmt_num(incumbency.get('p_val'))})")
        logistic = vis_results.get('logistic_close_victory') or {}
        if logistic:
            lines.append(f"Close victories (<{self.config.CLOSE_MARGIN_THRESHOLD}% margin): {logistic.get('close_victories', 0)} of {logistic.get('sample_size', 0)} ({fmt_ratio(logistic.get('close_share'))})")
            model = logistic.get('model')
            if model is not None:
                for param in ['ever_ran_as_challenger', 'mean_score', 'ever_contested_primary']:
                    if param in model.params:
                        lines.append(f"  {param.replace('_', ' ').title()}: β = {fmt_num(model.params[param])} (p = {fmt_num(model.pvalues[param])})")
        pathway = vis_results.get('vulnerability_pathway') or {}
        if pathway:
            lines.append(f"Progressive challengers: {fmt_ratio(pathway.get('progressive_challenger_rate'))} vs non-progressive {fmt_ratio(pathway.get('non_progressive_challenger_rate'))}")
            lines.append(f"Challenger margin difference (closest general): {fmt_num(pathway.get('challenger_margin_difference'))} pp")
            model = pathway.get('challenger_model')
            if model is not None and 'mean_score' in model.params:
                lines.append(f"  Logistic regression (challenger ~ ideology): β = {fmt_num(model.params['mean_score'])} (p = {fmt_num(model.pvalues['mean_score'])})")
        lines.append("")

        ideology = vis_results.get('contestation_by_ideology') or {}
        add_section("Contestation by Ideology")
        if ideology:
            counts = ideology.get('counts', {}) or {}
            lines.append("Counts: " + ", ".join([f"{k}={v}" for k, v in counts.items()]))
            primary_rates = ideology.get('primary_rates', {}) or {}
            general_rates = ideology.get('general_rates', {}) or {}
            lines.append(f"Primary contestation — Progressive: {fmt_ratio(primary_rates.get('Progressive'))} | Traditional: {fmt_ratio(primary_rates.get('Traditional'))}")
            chi_primary = ideology.get('chi_primary')
            if chi_primary:
                if primary_rates.get('Progressive') is not None and primary_rates.get('Traditional') is not None:
                    primary_diff = primary_rates.get('Progressive') - primary_rates.get('Traditional')
                    diff_text = fmt_ratio(primary_diff)
                else:
                    diff_text = "N/A"
                lines.append(f"  Difference: {diff_text} (χ² = {fmt_num(chi_primary[0])}, p = {fmt_num(chi_primary[1])})")
            lines.append(f"General contestation — Progressive: {fmt_ratio(general_rates.get('Progressive'))} | Traditional: {fmt_ratio(general_rates.get('Traditional'))}")
            chi_general = ideology.get('chi_general')
            if chi_general:
                if general_rates.get('Progressive') is not None and general_rates.get('Traditional') is not None:
                    general_diff = general_rates.get('Progressive') - general_rates.get('Traditional')
                    general_diff_text = fmt_ratio(general_diff)
                else:
                    general_diff_text = "N/A"
                lines.append(f"  Difference: {general_diff_text} (χ² = {fmt_num(chi_general[0])}, p = {fmt_num(chi_general[1])})")
        else:
            lines.append("Ideology contestation breakdown unavailable.")
        lines.append("")

        rater = vis_results.get('rater_position_scores') or {}
        add_section("Mean Ideology Ratings by Rater Position")
        means = rater.get('means', {}) or {}
        if means:
            lines.append(f"Prosecutors rating prosecutors: {fmt_num(means.get('Prosecutors rating prosecutors'))}")
            lines.append(f"Academics rating prosecutors: {fmt_num(means.get('Academics rating prosecutors'))}")
            lines.append(f"Defense attorneys rating prosecutors: {fmt_num(means.get('Defense attorneys rating prosecutors'))}")
            lines.append(f"Other professionals rating prosecutors: {fmt_num(means.get('Other professionals'))}")
            if rater.get('anova_f') is not None:
                lines.append(f"ANOVA F({fmt_num(rater.get('anova_df1'), 0)}, {fmt_num(rater.get('anova_df2'), 0)}) = {fmt_num(rater.get('anova_f'))}, p = {fmt_num(rater.get('anova_p'))}")
        else:
            lines.append("Rater position differences unavailable.")
        lines.append("")

        regional = vis_results.get('regional_patterns') or {}
        add_section("Regional Patterns")
        means = regional.get('means') or {}
        if means:
            for region, value in means.items():
                lines.append(f"{region}: {fmt_num(value)}")
            lines.append(f"ANOVA F({fmt_num(regional.get('anova_df1'), 0)}, {fmt_num(regional.get('anova_df2'), 0)}) = {fmt_num(regional.get('anova_f'))}, p = {fmt_num(regional.get('anova_p'))}")
        else:
            lines.append("Regional comparisons unavailable.")
        lines.append("")

        urban = vis_results.get('urban_rural') or {}
        add_section("Urban vs. Rural (Notable Prosecutors)")
        means = urban.get('means') or {}
        if means:
            for label, value in means.items():
                lines.append(f"{label}: {fmt_num(value)}")
            lines.append(f"Population correlation r = {fmt_num(urban.get('population_correlation'))} (p = {fmt_num(urban.get('population_corr_p'))})")
        else:
            lines.append("Urban/rural breakdown unavailable.")
        lines.append("")

        geo = vis_results.get('geographic_concentration') or {}
        add_section("Geographic Concentration of Progressive vs Traditional Prosecutors")
        if geo:
            lines.append(f"Progressive prosecutors (≥2.5): total {geo.get('progressive_total', 0)}, California share {geo.get('progressive_top10_ca', 0)} of top 10, major city count {geo.get('progressive_top10_major_city', 0)}, located in Biden 2020 states {geo.get('progressive_biden_states', 0)}")
            lines.append(f"Traditional prosecutors (<2.5): total {geo.get('traditional_total', 0)}, southern states {geo.get('traditional_southern_states', 0)}, located in Trump 2020 states {geo.get('traditional_trump_states', 0)}")
        else:
            lines.append("Geographic concentration metrics unavailable.")
        lines.append("")

        add_section("Appended Detailed Sections")
        append_targets = [
            ('CORRECTED_FINDINGS_SUMMARY.txt', Path(self.config.OUTPUT_DIR) / 'CORRECTED_FINDINGS_SUMMARY.txt'),
            ('INTEGRATED_MASTER_SUMMARY.txt', summary_path),
            ('EXTENDED_VISIBILITY_COMPETITION.txt', visibility_path),
            ('INTEGRATED_RESEARCH_QUESTIONS.txt', rq_path)
        ]
        for label, path in append_targets:
            if isinstance(path, (str, Path)):
                path_obj = Path(path)
            else:
                path_obj = None
            if path_obj and path_obj.exists():
                lines.append("")
                lines.append("=" * 90)
                lines.append(f"APPENDIX: {label}")
                lines.append("=" * 90)
                content = path_obj.read_text(encoding='utf-8').strip()
                lines.append(content)

        master_path = Path(self.config.OUTPUT_DIR) / 'MASTER_REPORT_FULL.txt'
        master_path.write_text("\n".join(lines), encoding='utf-8')
        print(f"✓ Unified master report saved -> {master_path}")
        return master_path

    def run_research_questions(self):
        print_section("RUNNING RESEARCH QUESTIONS (TIER 1)")
        dfm = getattr(self.analyzer, 'df_matched_all', pd.DataFrame())
        if dfm.empty:
            print("No matched election data available; skipping research questions.")
            return None

        lines = []
        lines.append("=" * 90)
        lines.append("TIER 1 RESEARCH QUESTIONS")
        lines.append("=" * 90)
        lines.append("")

        notable = dfm[dfm['is_notable']].copy()
        notable['progressive'] = notable['name'].isin(self.config.PROGRESSIVE_NAMES)
        notable['traditional'] = notable['name'].isin(self.config.TRADITIONAL_NAMES)

        lines.append("RQ1: Do progressive notable prosecutors face more contested elections?")
        lines.append("-" * 90)
        prog = notable[notable['progressive']]
        trad = notable[notable['traditional']]
        if len(prog) and len(trad):
            lines.append(f"Progressive (n={len(prog)}): contested any = {prog['ever_contested_any'].mean()*100:.1f}% | contested general = {prog['ever_contested_general'].mean()*100:.1f}%")
            lines.append(f"Traditional (n={len(trad)}): contested any = {trad['ever_contested_any'].mean()*100:.1f}% | contested general = {trad['ever_contested_general'].mean()*100:.1f}%")
            contingency = pd.crosstab(
                notable[notable['progressive'] | notable['traditional']]['progressive'],
                notable[notable['progressive'] | notable['traditional']]['ever_contested_general']
            )
            if contingency.shape == (2, 2):
                chi2, p_val, *_ = chi2_contingency(contingency)
                lines.append(f"Chi-square (contested general): χ²={chi2:.3f}, p={p_val:.4f}")
        else:
            lines.append("Insufficient overlap between progressive/traditional notable prosecutors for statistical comparison.")
        lines.append("")

        lines.append("RQ2: Recall context — margins for recalled prosecutors")
        lines.append("-" * 90)
        recalled = notable[notable['recalled'] == True]
        if len(recalled):
            for _, row in recalled.iterrows():
                lines.append(f"{row['name']}: closest primary margin={row['closest_primary_margin'] if pd.notna(row['closest_primary_margin']) else 'N/A'} | closest general margin={row['closest_general_margin'] if pd.notna(row['closest_general_margin']) else 'N/A'}")
        else:
            lines.append("No recalled prosecutors present in matched dataset.")
        lines.append("")

        lines.append("RQ3: Does familiarity differ for challengers vs incumbents?")
        lines.append("-" * 90)
        challengers = notable[notable['ever_ran_as_challenger'] == True]
        incumbents = notable[notable['ever_ran_as_incumbent'] == True]
        if len(challengers) and len(incumbents):
            lines.append(f"Challengers (n={len(challengers)}): mean familiarity={challengers['familiarity_rate'].mean():.2f}%")
            lines.append(f"Incumbents (n={len(incumbents)}): mean familiarity={incumbents['familiarity_rate'].mean():.2f}%")
            t_stat, p_val = ttest_ind(challengers['familiarity_rate'], incumbents['familiarity_rate'], equal_var=False, nan_policy='omit')
            lines.append(f"t-test: t={t_stat:.3f}, p={p_val:.4f}")
        else:
            lines.append("Insufficient challenger or incumbent data for comparison.")
        lines.append("")

        rq_path = Path(self.config.OUTPUT_DIR) / 'INTEGRATED_RESEARCH_QUESTIONS.txt'
        rq_path.write_text("\n".join(lines), encoding='utf-8')
        print(f"✓ Research questions summary saved -> {rq_path}")
        return rq_path

    def export_all(self):
        self.generate_visualizations()
        summary_path = self.produce_extended_summary()
        visibility_path = self.produce_visibility_competition_report()
        rq_path = self.run_research_questions()
        master_path = self.produce_unified_master_report(summary_path, visibility_path, rq_path)
        return summary_path, visibility_path, rq_path, master_path




# ================================================================================
# MAIN EXECUTION
# ================================================================================

def main():
    """Main execution function"""
    
    print("\n" + "="*80)
    print("CORRECTED PROSECUTOR IDEOLOGY SURVEY - COMPREHENSIVE ANALYSIS")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Survey file: {Config.SURVEY_FILE}")
    print(f"  Election file: {Config.ELECTION_FILE}")
    print(f"  Output directory: {Config.OUTPUT_DIR}")
    print(f"  Min ratings threshold: {Config.MIN_RATINGS_THRESHOLD}")
    print("\nCRITICAL CORRECTIONS:")
    print("  ✓ Notable prosecutor familiarity uses 407 (completed national section)")
    print("  ✓ State prosecutor familiarity uses state respondents who engaged")
    print("  ✓ Scale: 1-4 (Very Traditional to Very Progressive)")
    print("\nMETHODOLOGY:")
    print("  • Total initiated: 496")
    print("  • After consent: 493")
    print("  • Completed national section: 407 (82.1%)")
    print("  • Completed full survey: 365 (74.0%)")
    print("="*80)
    
    # Check if files exist
    if not Path(Config.SURVEY_FILE).exists():
        print(f"\n❌ ERROR: Survey file not found: {Config.SURVEY_FILE}")
        print("Please ensure the file is in the current directory.")
        return
    
    if not Path(Config.ELECTION_FILE).exists():
        print(f"\n❌ ERROR: Election file not found: {Config.ELECTION_FILE}")
        print("Please ensure the file is in the current directory.")
        return
    
    # Load data
    loader = DataLoader(Config.SURVEY_FILE, Config.ELECTION_FILE).load()
    
    # Run analyses
    analyzer = ProsecutorAnalyzer(loader)
    analyzer.build_prosecutor_dataset()
    analyzer.match_with_elections()
    analyzer.analyze_familiarity_patterns()
    analyzer.analyze_incumbency()
    analyzer.analyze_visibility_and_competition()
    analyzer.analyze_contestation()
    analyzer.analyze_recall_risk()
    analyzer.run_multivariate_models()
    analyzer.export_results(Config.OUTPUT_DIR)

    # Extended integrated outputs
    reporter = IntegratedReporter(analyzer)
    summary_path, visibility_path, rq_path, master_path = reporter.export_all()
    print(f"\nExtended summary: {summary_path}")
    if visibility_path:
        print(f"Visibility & competition report: {visibility_path}")
    if rq_path:
        print(f"Research questions report: {rq_path}")
    if master_path:
        print(f"Unified master report: {master_path}")

    print_section("✅ CORRECTED ANALYSIS COMPLETE!", "=")
    print(f"All outputs saved to: {Config.OUTPUT_DIR}/")
    print("\nGenerated files:")
    print("  📊 Data Files:")
    print("     - all_prosecutors_CORRECTED.csv")
    print("     - prosecutors_filtered_ideology_CORRECTED.csv")
    print("     - matched_prosecutors_FULL_CORRECTED.csv")
    print("     - matched_prosecutors_FILTERED_CORRECTED.csv")
    print("     - CORRECTED_FINDINGS_SUMMARY.txt")
    print("     - EXTENDED_VISIBILITY_COMPETITION.txt")
    print("     - INTEGRATED_MASTER_SUMMARY.txt")
    print("     - INTEGRATED_RESEARCH_QUESTIONS.txt")
    print("     - MASTER_REPORT_FULL.txt")
    print("\n" + "="*80)
    print("\n🎯 NEXT STEPS:")
    print("1. Review CORRECTED_FINDINGS_SUMMARY.txt for key changes")
    print("2. Compare with old report to identify what needs updating")
    print("3. Pay special attention to state prosecutor familiarity rates!")
    print("="*80)


if __name__ == '__main__':
    main()

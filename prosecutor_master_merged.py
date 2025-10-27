#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
PROSECUTOR SURVEY × ELECTIONS — MASTER MERGED SCRIPT
================================================================================
Purpose
-------
This single script merges the **best parts** from four prior versions:

1) Robust cleaning & name/jurisdiction **matching utilities** (new script).
2) Corrected **familiarity denominators** and survey loader (corrected master).
3) **Comprehensive analysis** blocks (rankings, transitions, state patterns).
4) A compact suite of **Tier 1 research questions** for quick checks.

How to run
----------
python prosecutor_master_merged.py

Expected inputs (adjust in Config if needed):
- SURVEY_FILE: Qualtrics CSV (with header row + labels row).
- ELECTION_FILE: Elections CSV (reconciled).
Outputs:
- outputs/corrected/ ... CSVs and images

Author: Dvir Yogev (BERQ-J) — merged by helper
Date: 2025-10-24
================================================================================
"""

import os
import re
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd

# Optional plotting/stats only used if available
try:
    import matplotlib.pyplot as plt
    HAVE_PLT = True
except Exception:
    HAVE_PLT = False

try:
    import seaborn as sns
    HAVE_SNS = True
except Exception:
    HAVE_SNS = False

try:
    from scipy import stats
    from scipy.stats import pearsonr, spearmanr, ttest_ind, mannwhitneyu, chi2_contingency
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    HAVE_SM = True
except Exception:
    HAVE_SM = False


# =============================================================================
# CONFIG
# =============================================================================

class Config:
    # Files
    SURVEY_FILE = 'PP+survey+draft_October+16,+2025_13.32.csv'
    ELECTION_FILE = 'elections_with_reconciled_contested.csv'

    # Output dirs
    OUTPUT_DIR = Path('outputs/corrected')
    VIZ_DIR = OUTPUT_DIR / 'visualizations'
    TABLE_DIR = OUTPUT_DIR / 'tables'

    # Familiarity threshold for "reliable"
    MIN_RATINGS_THRESHOLD = 10

    # Close election threshold in % points
    CLOSE_MARGIN_THRESHOLD = 55  # e.g., <=55% winner share counted "close"

    # Correct 1–4 ideology scale
    RATING_MAP = {
        'Very Traditional': 1,
        'Traditional': 2,
        'Not Familiar': np.nan,
        'Progressive': 3,
        'Very Progressive': 4,
    }


# =============================================================================
# UTILITIES — (from the "new" script, simplified)
# =============================================================================

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


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

def std_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().str.strip().str.replace(r'\s+', ' ', regex=True)

def extract_state_from_jurisdiction(juris_str: str) -> Optional[str]:
    if not isinstance(juris_str, str):
        return None
    parts = [p.strip() for p in juris_str.split(',')]
    if len(parts) >= 2:
        state_abbrev = parts[-1].upper()
        return STATE_ABBREV_TO_NAME.get(state_abbrev, state_abbrev)
    return None

def clean_jurisdiction_for_matching(juris_str: str) -> str:
    if not isinstance(juris_str, str):
        return ''
    if ',' in juris_str:
        juris_str = juris_str.rsplit(',', 1)[0]
    juris_str = re.sub(r'\s*\([^)]*\)', '', juris_str)       # drop parenthetical
    juris_str = re.sub(r'\bCounty\b', '', juris_str, flags=re.IGNORECASE)
    juris_str = re.sub(r'\s+', ' ', juris_str).strip()
    return juris_str

def norm_first(s: str) -> str:
    if not isinstance(s, str) or not s.strip():
        return ''
    s = s.lower().strip()
    return NICKNAME_MAP.get(s, s)

def first_name_compatible(a: str, b: str) -> bool:
    a = norm_first(a); b = norm_first(b)
    if not a or not b:
        return True
    if a == b:
        return True
    if a.startswith(b) or b.startswith(a):
        return True
    return a[0] == b[0]


# =============================================================================
# DATA LOADER with CORRECTED FAMILIARITY DENOMINATORS
# =============================================================================

class DataLoader:
    """
    Loads the Qualtrics survey and elections, and builds per‑prosecutor
    summary tables with **correct familiarity denominators**.
    """

    def __init__(self, survey_file: str, election_file: str):
        self.survey_file = Path(survey_file)
        self.election_file = Path(election_file)

    def load(self):
        # Load survey (Qualtrics: skip the 2nd row with labels)
        try:
            self.df = pd.read_csv(self.survey_file, skiprows=[1])
            self.questions_df = pd.read_csv(self.survey_file, nrows=1)
            print(f"✓ Loaded survey ({len(self.df)}) with Qualtrics header")
        except Exception:
            self.df = pd.read_csv(self.survey_file)
            self.questions_df = None
            print(f"✓ Loaded survey ({len(self.df)}) in standard CSV format")

        # Consent filter
        if 'con' in self.df.columns:
            before = len(self.df)
            self.df = self.df[self.df['con'] == 'Agree'].copy()
            print(f"✓ Consent filter: {before} → {len(self.df)}")

        # Determine "completed national section" via 'more' sentinel
        if 'more' in self.df.columns:
            self.completed_national_section = int(self.df['more'].notna().sum())
        else:
            self.completed_national_section = len(self.df)
        print(f"✓ Completed national section: {self.completed_national_section}")

        # Election data
        try:
            self.elec = pd.read_csv(self.election_file)
            print(f"✓ Loaded elections ({len(self.elec)})")
        except Exception as e:
            print(f"✗ Elections load failed: {e}")
            self.elec = pd.DataFrame()

        # Identify notable and state DA columns
        self.notable_cols = [c for c in self.df.columns if c.startswith('notable_')]

        # Heuristic to find state DA columns (State_# style that actually hold ratings)
        self.state_da_cols = []
        for col in self.df.columns:
            if '_' in col and not col.startswith('Q') and not col.startswith('notable'):
                parts = col.split('_')
                if len(parts) == 2 and parts[0] not in ['gender', 'race', 'Q55']:
                    if self.df[col].dropna().astype(str).str.contains('Progressive|Traditional|Not Familiar').any():
                        self.state_da_cols.append(col)

        print(f"✓ Found {len(self.notable_cols)} notable columns; {len(self.state_da_cols)} state DA columns")
        return self

    # --- helpers ----------------------------------------------------------------

    def _parse_name_from_question(self, q_id: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Attempt to parse LNAME, FNAME, JURIS from the Qualtrics questions header row.
        Returns (fname, lname, jurisdiction) — any may be None.
        """
        if self.questions_df is None or q_id not in self.questions_df.columns:
            return (None, None, None)
        qtext = str(self.questions_df[q_id].iloc[0])
        tail = qtext.split(' - ')[-1] if ' - ' in qtext else qtext
        parts = [p.strip() for p in tail.split('\t') if p.strip()]
        if len(parts) >= 3:
            lname, fname, juris = parts[0], parts[1], parts[2]
            return (fname, lname, juris)
        if len(parts) == 2:
            lname, fname = parts[0], parts[1]
            return (fname, lname, None)
        return (None, None, None)

    # --- builders ---------------------------------------------------------------

    def build_notable_table(self) -> pd.DataFrame:
        """
        Notable prosecutors: familiarity denominator = people who reached end of
        national section (completed_national_section). Only 1–4 ratings counted
        as "substantive".
        """
        rows = []
        for col in self.notable_cols:
            s = self.df[col]
            numeric = s.map(Config.RATING_MAP).dropna()
            if numeric.empty:
                continue
            # Extract name & jurisdiction from header mapping if available
            fname, lname, juris = self._parse_name_from_question(col)
            name = None
            if fname or lname:
                name = (' '.join([x for x in [fname, lname] if x])).strip()
            else:
                # Fallback to column label
                name = col

            rows.append({
                'column': col,
                'name': name,
                'fname': fname,
                'lname': lname,
                'jurisdiction': juris,
                'jurisdiction_clean': clean_jurisdiction_for_matching(juris or ''),
                'state_name': extract_state_from_jurisdiction(juris or ''),
                'substantive_ratings': len(numeric),
                'total_responses': self.completed_national_section,
                'familiarity_rate': (len(numeric) / max(1, self.completed_national_section)) * 100.0,
                'mean_score': numeric.mean(),
                'median_score': numeric.median(),
                'std_score': numeric.std(),
                'is_notable': True
            })
        df = pd.DataFrame(rows)
        return df

    def build_state_table(self) -> pd.DataFrame:
        """
        State prosecutors: denominator = **respondents from that state who
        engaged with their state section** (responded to at least one state DA).
        """
        # map state -> list of columns
        state_to_cols: Dict[str, List[str]] = {}
        for col in self.state_da_cols:
            st = col.split('_', 1)[0]
            state_to_cols.setdefault(st, []).append(col)

        # who engaged per state
        engaged_counts: Dict[str, int] = {st: 0 for st in state_to_cols}
        # Build fast mask per respondent
        for st, cols in state_to_cols.items():
            mask = self.df[cols].notna().any(axis=1)
            engaged_counts[st] = int(mask.sum())

        rows = []
        for col in self.state_da_cols:
            st = col.split('_', 1)[0]
            s = self.df[col]
            numeric = s.map(Config.RATING_MAP).dropna()
            if numeric.empty:
                continue

            fname, lname, juris = self._parse_name_from_question(col)
            name = (' '.join([x for x in [fname, lname] if x])).strip() if (fname or lname) else col

            rows.append({
                'column': col,
                'name': name,
                'fname': fname,
                'lname': lname,
                'jurisdiction': juris,
                'jurisdiction_clean': clean_jurisdiction_for_matching(juris or ''),
                'state_name': st,
                'substantive_ratings': len(numeric),
                'total_responses': engaged_counts.get(st, 0),
                'familiarity_rate': (len(numeric) / max(1, engaged_counts.get(st, 0))) * 100.0,
                'mean_score': numeric.mean(),
                'median_score': numeric.median(),
                'std_score': numeric.std(),
                'is_notable': False
            })
        df = pd.DataFrame(rows)
        return df


# =============================================================================
# ELECTION MATCHING — robust jurisdiction + name logic
# =============================================================================

class ElectionMatcher:
    """
    Performs tolerant matching of survey DA rows to election rows using:
    - cleaned jurisdiction string
    - last name + compatible first name (nickname/initial aware)
    """

    def __init__(self, elections_df: pd.DataFrame):
        self.e = elections_df.copy()
        # Derive cleaned keys if present
        for col in ['jurisdiction', 'office_jurisdiction', 'county', 'juris']:
            if col in self.e.columns:
                self.e['jurisdiction_clean'] = self.e[col].astype(str).map(clean_jurisdiction_for_matching)
                break
        if 'jurisdiction_clean' not in self.e.columns:
            self.e['jurisdiction_clean'] = self.e.iloc[:, 0].astype(str).map(clean_jurisdiction_for_matching)

        # Normalize first/last names if available
        for c in ['first', 'firstname', 'fname']:
            if c in self.e.columns:
                self.e['e_first'] = self.e[c].astype(str)
                break
        if 'e_first' not in self.e.columns:
            self.e['e_first'] = ''

        for c in ['last', 'lastname', 'lname', 'surname']:
            if c in self.e.columns:
                self.e['e_last'] = self.e[c].astype(str).str.lower().str.strip()
                break
        if 'e_last' not in self.e.columns:
            self.e['e_last'] = ''

    def match(self, df_da: pd.DataFrame) -> pd.DataFrame:
        df = df_da.copy()
        df['lname_key'] = df['lname'].astype(str).str.lower().str.strip()
        df['jurisdiction_clean'] = df['jurisdiction_clean'].astype(str)

        # First, jurisdiction-clean inner join (fast narrow)
        merged = df.merge(
            self.e,
            on='jurisdiction_clean',
            how='left',
            suffixes=('', '_elec')
        )

        # Then filter by last name match + compatible first name (or missing)
        def row_ok(r):
            if pd.isna(r.get('e_last', '')):
                return False
            if r['lname_key'] and str(r['e_last']).lower().strip() == r['lname_key']:
                return first_name_compatible(str(r.get('fname', '')), str(r.get('e_first', '')))
            # If we can't compare last names, keep row but downweight later
            return False

        if not merged.empty:
            mask = merged.apply(row_ok, axis=1)
            filtered = merged[mask].copy()
            if filtered.empty:
                # fall back to jurisdiction-only (may be many-to-many)
                filtered = merged.copy()
        else:
            filtered = df.copy()

        # Deduplicate to one row per DA by preferring rows with name match
        def rank_row(r):
            score = 0
            if str(r.get('e_last', '')).lower().strip() == r.get('lname_key', ''):
                score += 2
            if first_name_compatible(str(r.get('fname', '')), str(r.get('e_first', ''))):
                score += 1
            return score

        if not filtered.empty:
            filtered['match_score'] = filtered.apply(rank_row, axis=1)
            filtered = (filtered.sort_values(['column', 'match_score'], ascending=[True, False])
                               .drop_duplicates(subset=['column'], keep='first'))

        return filtered


# =============================================================================
# ANALYSIS — (comprehensive, compact summaries in prints/CSVs)
# =============================================================================

class Analyzer:
    def __init__(self, df_notable: pd.DataFrame, df_state: pd.DataFrame):
        self.notable = df_notable.copy()
        self.state = df_state.copy()

    def save_csvs(self):
        ensure_dir(Config.OUTPUT_DIR)
        self.notable.to_csv(Config.OUTPUT_DIR / 'notable_prosecutors_summary.csv', index=False)
        self.state.to_csv(Config.OUTPUT_DIR / 'state_prosecutors_summary.csv', index=False)

    def rankings(self):
        """Top/bottom by ideology mean among sufficiently rated DAs (notable only)."""
        df = self.notable.copy()
        rel = df[df['substantive_ratings'] >= Config.MIN_RATINGS_THRESHOLD].copy()
        if rel.empty:
            print("No notable DAs with sufficient ratings.")
            return
        rel = rel.sort_values('mean_score', ascending=False)

        print("\n🏆 MOST PROGRESSIVE (mean ≥):")
        for i, r in enumerate(rel.head(5).itertuples(), 1):
            print(f"  {i}. {r.name} — {r.mean_score:.2f} (N={r.substantive_ratings})")

        print("\n🏛️  MOST TRADITIONAL:")
        for i, r in enumerate(rel.tail(5).sort_values('mean_score').itertuples(), 1):
            print(f"  {i}. {r.name} — {r.mean_score:.2f} (N={r.substantive_ratings})")

        # Save
        ensure_dir(Config.TABLE_DIR)
        rel.to_csv(Config.TABLE_DIR / 'notable_rankings.csv', index=False)

    def transitions(self):
        """Jurisdiction predecessor→successor shifts (requires both present)."""
        # Build simple key: "<jurisdiction_clean> – <lname>"
        def key(df):
            return df['jurisdiction_clean'].astype(str) + '—' + df['lname'].astype(str).str.lower().str.strip()

        N = self.notable.copy()
        N['key'] = key(N)

        # Hand‑entered known transitions can be listed here if desired; we also try to infer via same jurisdiction & different last name.
        # For now, we require the user to provide a list if they need exact pairs; we save the base table for plotting externally.
        ensure_dir(Config.TABLE_DIR)
        N[['name','jurisdiction','jurisdiction_clean','mean_score','substantive_ratings']].to_csv(
            Config.TABLE_DIR / 'notable_for_transitions.csv', index=False
        )
        print("\n✓ Wrote base transitions table (provide explicit pairs to plot direction arrows).")

    # --- Tier 1 RQs (compact) -------------------------------------------------

    def tier1(self, merged_notable: pd.DataFrame, merged_state: pd.DataFrame):
        """
        Run compact Tier‑1 style checks if the merged data include contestation
        and margin fields (ever_contested_*, closest_*_margin, etc.).
        """
        if merged_notable is None or merged_notable.empty:
            print("\n(Tier1) No merged notable data available.")
            return

        print("\n================= TIER 1 QUICK CHECKS =================")

        # RQ: Familiarity vs number of elections
        if 'num_elections' in merged_notable.columns:
            if HAVE_SCIPY and merged_notable['num_elections'].notna().any():
                try:
                    r, p = pearsonr(merged_notable['num_elections'].fillna(0), merged_notable['familiarity_rate'].fillna(0))
                    print(f"[RQ] num_elections ↔ familiarity_rate: r={r:.3f}, p={p:.4f}")
                except Exception as e:
                    print(f"[RQ] Correlation failed: {e}")

        # RQ: Challengers vs incumbents familiarity
        if {'ever_ran_as_challenger','ever_ran_as_incumbent'}.issubset(merged_notable.columns):
            ch = merged_notable[merged_notable['ever_ran_as_challenger']]
            inc = merged_notable[merged_notable['ever_ran_as_incumbent']]
            if not ch.empty and not inc.empty:
                if HAVE_SCIPY:
                    t, p = ttest_ind(ch['familiarity_rate'], inc['familiarity_rate'], equal_var=False, nan_policy='omit')
                    print(f"[RQ] Challenger vs Incumbent familiarity: t={t:.3f}, p={p:.4f}")
                print(f"     Challenger mean={np.nanmean(ch['familiarity_rate']):.2f}%, Incumbent mean={np.nanmean(inc['familiarity_rate']):.2f}%")

        # RQ: Recalled vs others close margins (if available)
        if {'recalled','closest_general_margin'}.issubset(merged_notable.columns):
            rec = merged_notable[merged_notable['recalled']]
            non = merged_notable[~merged_notable['recalled']]
            print(f"[RQ] Recalled mean closest_general_margin: {np.nanmean(rec['closest_general_margin']):.1f}% | others: {np.nanmean(non['closest_general_margin']):.1f}%")

        print("========================================================\n")


# =============================================================================
# MAIN
# =============================================================================

def main():
    ensure_dir(Config.OUTPUT_DIR); ensure_dir(Config.VIZ_DIR); ensure_dir(Config.TABLE_DIR)

    loader = DataLoader(Config.SURVEY_FILE, Config.ELECTION_FILE).load()
    df_notable = loader.build_notable_table()
    df_state = loader.build_state_table()

    # Save the base summaries
    analyzer = Analyzer(df_notable, df_state)
    analyzer.save_csvs()
    analyzer.rankings()
    analyzer.transitions()

    # Merge to elections if file present
    merged_notable = None; merged_state = None
    if not loader.elec.empty:
        matcher = ElectionMatcher(loader.elec)
        merged_notable = matcher.match(df_notable)
        merged_state = matcher.match(df_state)

        # Persist merged tables for downstream Tier‑1 style analyses
        merged_out_dir = Config.OUTPUT_DIR
        merged_notable.to_csv(merged_out_dir / 'notable_prosecutors_elections.csv', index=False)
        merged_state.to_csv(merged_out_dir / 'state_prosecutors_elections.csv', index=False)
        print(f"\n✓ Wrote merged tables → {merged_out_dir}")

    # Optional Tier‑1 questions (if merge had the relevant cols)
    analyzer.tier1(merged_notable, merged_state)

if __name__ == '__main__':
    main()

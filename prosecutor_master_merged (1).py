#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
PROSECUTOR SURVEY × ELECTIONS — MASTER MERGED SCRIPT (with Plot Pack)
================================================================================
Purpose
-------
Merges the **best parts** from prior versions and adds a one-shot plot pack:

1) Robust cleaning & name/jurisdiction **matching utilities** (new script).
2) Corrected **familiarity denominators** and survey loader (corrected master).
3) **Comprehensive analysis** blocks (rankings, transitions scaffolding).
4) **Tier 1** quick checks (contestations/recalls vs familiarity).
5) NEW: **Plot pack** — rankings, familiarity histograms, pinned transitions
   plotted as arrows with deltas.

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

# Only matplotlib is used for charts (per policy)
import matplotlib.pyplot as plt

try:
    from scipy.stats import pearsonr, ttest_ind
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


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

    # --- PINNED TRANSITIONS (edit/add freely) ---------------------------------
    # Matching uses case-insensitive substring on jurisdiction_clean
    # and last-name matching (with tolerant O'/’ handling).
    TRANSITIONS = [
        {
            'jurisdiction_contains': 'san francisco',
            'from_name': 'Chesa Boudin',
            'to_name': 'Brooke Jenkins'
        },
        {
            'jurisdiction_contains': 'alameda',
            'from_name': "Nancy O’Malley",   # smart apostrophe okay
            'to_name': 'Pamela Price'
        },
        # Add more here as needed...
        # {
        #     'jurisdiction_contains': 'los angeles',
        #     'from_name': 'George Gascon',
        #     'to_name': 'Nathan Hochman'
        # },
    ]


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

def last_from_fullname(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        return ''
    # normalize straight/smart apostrophes to same form, strip non-letters except apostrophes/dashes
    s = re.sub(r'[’]', "'", name.strip())
    parts = s.split()
    if not parts:
        return ''
    return re.sub(r"[^A-Za-z'\\-]", '', parts[-1]).lower()

def first_from_fullname(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        return ''
    s = re.sub(r'[’]', "'", name.strip())
    parts = s.split()
    if not parts:
        return ''
    return re.sub(r"[^A-Za-z'\\-]", '', ' '.join(parts[:-1])).lower()


# =============================================================================
# DATA LOADER with CORRECTED FAMILIARITY DENOMINATORS
# =============================================================================

class DataLoader:
    """
    Loads the Qualtrics survey and elections, and builds per-prosecutor
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
        Robustly parse FNAME, LNAME, JURIS from Qualtrics question label row.
        Works with patterns like:
          "... - Larry Krasner, Philadelphia, PA"
          "... - Kim Foxx, Cook County (Chicago), IL"
        Returns (fname, lname, jurisdiction). Any may be None.
        """
        if self.questions_df is None or q_id not in self.questions_df.columns:
            return (None, None, None)
        qtext = str(self.questions_df[q_id].iloc[0])
        if ' - ' in qtext:
            tail = qtext.rsplit(' - ', 1)[-1].strip()
        else:
            tail = qtext.strip()

        # Split by the first comma: "Full Name, Jurisdiction..."
        if ',' in tail:
            name_part, juris_part = tail.split(',', 1)
            name_part = name_part.strip()
            juris = juris_part.strip().strip(',')
        else:
            name_part = tail
            juris = None

        # Derive first/last from the name_part (last token = last name)
        fname = first_from_fullname(name_part)
        lname = last_from_fullname(name_part)
        # Keep the pretty cased name if we want to display later
        return (fname if fname else None, lname if lname else None, juris if juris else None)

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
            # Construct display name if possible
            if fname or lname:
                disp_name = ' '.join([x for x in [fname, lname] if x]).title()
            else:
                disp_name = col

            rows.append({
                'column': col,
                'name': disp_name,
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
            disp_name = ' '.join([x for x in [fname, lname] if x]).title() if (fname or lname) else col

            rows.append({
                'column': col,
                'name': disp_name,
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

        merged = df.merge(
            self.e,
            on='jurisdiction_clean',
            how='left',
            suffixes=('', '_elec')
        )

        def row_ok(r):
            if pd.isna(r.get('e_last', '')):
                return False
            if r['lname_key'] and str(r['e_last']).lower().strip() == r['lname_key']:
                return first_name_compatible(str(r.get('fname', '')), str(r.get('e_first', '')))
            return False

        if not merged.empty:
            mask = merged.apply(row_ok, axis=1)
            filtered = merged[mask].copy()
            if filtered.empty:
                filtered = merged.copy()
        else:
            filtered = df.copy()

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
# ANALYSIS + PLOTS
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

    def transitions_table(self):
        """Write base table for transitions (one row per notable)."""
        ensure_dir(Config.TABLE_DIR)
        self.notable[['name','fname','lname','jurisdiction','jurisdiction_clean','mean_score','substantive_ratings']].to_csv(
            Config.TABLE_DIR / 'notable_for_transitions.csv', index=False
        )

    # --- Tier 1 RQs (compact) -------------------------------------------------

    def tier1(self, merged_notable: pd.DataFrame, merged_state: pd.DataFrame):
        if merged_notable is None or merged_notable.empty:
            print("\n(Tier1) No merged notable data available.")
            return

        print("\n================= TIER 1 QUICK CHECKS =================")

        if 'num_elections' in merged_notable.columns:
            if HAVE_SCIPY and merged_notable['num_elections'].notna().any():
                try:
                    r, p = pearsonr(merged_notable['num_elections'].fillna(0), merged_notable['familiarity_rate'].fillna(0))
                    print(f"[RQ] num_elections ↔ familiarity_rate: r={r:.3f}, p={p:.4f}")
                except Exception as e:
                    print(f"[RQ] Correlation failed: {e}")

        if {'ever_ran_as_challenger','ever_ran_as_incumbent'}.issubset(merged_notable.columns):
            ch = merged_notable[merged_notable['ever_ran_as_challenger']]
            inc = merged_notable[merged_notable['ever_ran_as_incumbent']]
            if not ch.empty and not inc.empty and HAVE_SCIPY:
                t, p = ttest_ind(ch['familiarity_rate'], inc['familiarity_rate'], equal_var=False, nan_policy='omit')
                print(f"[RQ] Challenger vs Incumbent familiarity: t={t:.3f}, p={p:.4f}")
                print(f"     Challenger mean={np.nanmean(ch['familiarity_rate']):.2f}%, Incumbent mean={np.nanmean(inc['familiarity_rate']):.2f}%")

        if {'recalled','closest_general_margin'}.issubset(merged_notable.columns):
            rec = merged_notable[merged_notable['recalled']]
            non = merged_notable[~merged_notable['recalled']]
            print(f"[RQ] Recalled mean closest_general_margin: {np.nanmean(rec['closest_general_margin']):.1f}% | others: {np.nanmean(non['closest_general_margin']):.1f}%")

        print("========================================================\n")

    # --- Plot pack -------------------------------------------------------------

    def plot_rankings(self):
        rel = self.notable[self.notable['substantive_ratings'] >= Config.MIN_RATINGS_THRESHOLD].copy()
        if rel.empty:
            return
        rel = rel.sort_values('mean_score', ascending=False)

        ensure_dir(Config.VIZ_DIR)

        # Top 10 progressive
        top = rel.head(10)
        plt.figure()
        plt.barh(top['name'], top['mean_score'])
        plt.gca().invert_yaxis()
        plt.xlabel('Mean Ideology Score (4=Very Progressive)')
        plt.title('Top 10 Progressive (Notables)')
        plt.tight_layout()
        plt.savefig(Config.VIZ_DIR / 'rankings_top10_progressive.png', dpi=200)
        plt.close()

        # Bottom 10 traditional
        bot = rel.tail(10).sort_values('mean_score')
        plt.figure()
        plt.barh(bot['name'], bot['mean_score'])
        plt.gca().invert_yaxis()
        plt.xlabel('Mean Ideology Score (1=Very Traditional)')
        plt.title('Bottom 10 (Most Traditional) — Notables')
        plt.tight_layout()
        plt.savefig(Config.VIZ_DIR / 'rankings_bottom10_traditional.png', dpi=200)
        plt.close()

    def plot_familiarity_histos(self):
        ensure_dir(Config.VIZ_DIR)

        # Notables familiarity histogram
        if not self.notable.empty and self.notable['familiarity_rate'].notna().any():
            plt.figure()
            vals = self.notable['familiarity_rate'].dropna().values
            plt.hist(vals, bins=20)
            plt.xlabel('Familiarity Rate (%)')
            plt.ylabel('Count of Notables')
            plt.title('Familiarity Distribution — Notables')
            plt.tight_layout()
            plt.savefig(Config.VIZ_DIR / 'familiarity_notables_hist.png', dpi=200)
            plt.close()

        # States familiarity histogram
        if not self.state.empty and self.state['familiarity_rate'].notna().any():
            plt.figure()
            vals = self.state['familiarity_rate'].dropna().values
            plt.hist(vals, bins=20)
            plt.xlabel('Familiarity Rate (%)')
            plt.ylabel('Count of State DAs')
            plt.title('Familiarity Distribution — State DAs')
            plt.tight_layout()
            plt.savefig(Config.VIZ_DIR / 'familiarity_states_hist.png', dpi=200)
            plt.close()

    def plot_transitions_arrows(self):
        ensure_dir(Config.VIZ_DIR)
        pinned = Config.TRANSITIONS
        rows = []
        for t in pinned:
            juris_sub = str(t['jurisdiction_contains']).lower()
            from_ln = last_from_fullname(t['from_name'])
            to_ln = last_from_fullname(t['to_name'])

            # Filter candidates by jurisdiction substring
            df = self.notable.copy()
            df['jc'] = df['jurisdiction_clean'].astype(str).str.lower()
            cand = df[df['jc'].str.contains(juris_sub, na=False)].copy()

            # Match by last name (tolerant of apostrophes)
            cand['lname_norm'] = df['lname'].astype(str).str.replace('’',"'").str.lower().str.strip()
            m_from = cand[cand['lname_norm'] == from_ln]
            m_to = cand[cand['lname_norm'] == to_ln]

            if m_from.empty or m_to.empty:
                # record a miss for transparency
                rows.append({
                    'jurisdiction_contains': juris_sub,
                    'from_name': t['from_name'],
                    'to_name': t['to_name'],
                    'status': 'NOT FOUND',
                    'from_mean': np.nan,
                    'to_mean': np.nan,
                    'delta': np.nan,
                    'figure': ''
                })
                continue

            f = m_from.iloc[0]
            g = m_to.iloc[0]
            y0 = float(f['mean_score'])
            y1 = float(g['mean_score'])
            delta = y1 - y0

            # Plot arrow
            plt.figure()
            plt.plot([0,1], [y0,y1], marker='o')
            plt.xticks([0,1], [t['from_name'], t['to_name']], rotation=0)
            plt.ylabel('Mean Ideology Score (1–4)')
            title = f"Transition: {t['from_name']} → {t['to_name']} ({f['jurisdiction_clean']})"
            plt.title(title)
            plt.ylim(1,4)
            plt.grid(True, axis='y', linestyle='--', alpha=0.5)
            outf = Config.VIZ_DIR / f"transition_{from_ln}_to_{to_ln}.png"
            plt.tight_layout()
            plt.savefig(outf, dpi=200)
            plt.close()

            rows.append({
                'jurisdiction_contains': juris_sub,
                'from_name': t['from_name'],
                'to_name': t['to_name'],
                'status': 'OK',
                'from_mean': y0,
                'to_mean': y1,
                'delta': delta,
                'figure': str(outf)
            })

        # Save a small audit table
        audit = pd.DataFrame(rows)
        audit.to_csv(Config.TABLE_DIR / 'transitions_pinned_audit.csv', index=False)


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
    analyzer.transitions_table()

    # Merge to elections if file present
    merged_notable = None; merged_state = None
    if not loader.elec.empty:
        matcher = ElectionMatcher(loader.elec)
        merged_notable = matcher.match(df_notable)
        merged_state = matcher.match(df_state)

        # Persist merged tables for downstream Tier-1 style analyses
        merged_out_dir = Config.OUTPUT_DIR
        merged_notable.to_csv(merged_out_dir / 'notable_prosecutors_elections.csv', index=False)
        merged_state.to_csv(merged_out_dir / 'state_prosecutors_elections.csv', index=False)
        print(f"\n✓ Wrote merged tables → {merged_out_dir}")

    # Tier-1 quick checks
    analyzer.tier1(merged_notable, merged_state)

    # --- Plot pack
    analyzer.plot_rankings()
    analyzer.plot_familiarity_histos()
    analyzer.plot_transitions_arrows()
    print(f"✓ Plot pack written to: {Config.VIZ_DIR}")

if __name__ == '__main__':
    main()

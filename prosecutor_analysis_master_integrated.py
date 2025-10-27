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
        
    def analyze_prosecutor_column(self, col, col_name=None, state_name=None):
        """
        Analyze a single prosecutor column with CORRECTED familiarity calculation
        
        Parameters:
        -----------
        col : str
            Column name in the survey
        col_name : str
            Human-readable prosecutor name
        state_name : str
            State name (for state prosecutors only) - used to get correct denominator
        """
        if col_name is None:
            col_name = col
        
        responses = self.df_survey[col].dropna()
        
        # CRITICAL CORRECTION: Determine correct denominator
        if state_name and state_name in self.state_respondent_counts:
            # State prosecutor: use only respondents from that state who engaged with state section
            # "Engaged" means they responded to at least one state prosecutor from their state
            state_mask = self.df_survey['RespondentState'] == state_name
            state_respondents = self.df_survey[state_mask]
            
            # Find all state prosecutor columns for this state
            state_cols_for_this_state = [c for c in self.state_da_cols if c.startswith(state_name + '_')]
            
            # Count how many state respondents engaged with state prosecutors
            engaged_count = 0
            for idx in state_respondents.index:
                if any(pd.notna(self.df_survey.loc[idx, c]) for c in state_cols_for_this_state):
                    engaged_count += 1
            
            total_responses = engaged_count if engaged_count > 0 else self.state_respondent_counts[state_name]
        else:
            # Notable prosecutor: use completed_national_section (those who reached end of national section)
            total_responses = self.completed_national_section
        
        # Convert to numeric
        numeric_ratings = responses.map(Config.RATING_MAP)
        substantive_ratings = numeric_ratings.dropna()
        
        if len(substantive_ratings) == 0:
            return None
        
        # CORRECTED familiarity rate calculation
        familiarity_rate = (len(substantive_ratings) / total_responses) * 100 if total_responses > 0 else 0
        
        result = {
            'column': col,
            'name': col_name,
            'state': state_name if state_name else 'National',
            'total_responses': total_responses,
            'substantive_ratings': len(substantive_ratings),
            'familiarity_rate': familiarity_rate,
            'mean_score': substantive_ratings.mean(),
            'median_score': substantive_ratings.median(),
            'std_score': substantive_ratings.std(),
        }
        
        return result
    
    def build_prosecutor_dataset(self):
        """Build comprehensive prosecutor dataset with CORRECTED familiarity"""
        print_section("BUILDING PROSECUTOR DATASET (CORRECTED FAMILIARITY)")
        
        all_prosecutors = []
        
        # Process notable prosecutors (all respondents as denominator)
        print("Processing notable prosecutors...")
        for col in self.notable_cols:
            if col in Config.DA_NAMES:
                name, location = Config.DA_NAMES[col]
                result = self.analyze_prosecutor_column(col, name, state_name=None)
                if result:
                    result['location'] = location
                    result['is_notable'] = True
                    result['name'] = name
                    all_prosecutors.append(result)
        
        print(f"✓ Processed {len([p for p in all_prosecutors if p['is_notable']])} notable prosecutors")
        
        # Process state prosecutors (state-specific denominators)
        print("\nProcessing state prosecutors with CORRECTED denominators...")
        state_counts = {}
        
        for col in self.state_da_cols:
            # Extract state from column name (e.g., "California_1" -> "California")
            state_name = col.split('_')[0]
            
            # Get prosecutor name from question text
            if self.questions_df is not None and col in self.questions_df.columns:
                question_text = self.questions_df[col].iloc[0]
                if ' - ' in question_text:
                    name = question_text.split(' - ')[-1].strip()
                    
                    # CRITICAL: Pass state_name to get correct denominator
                    result = self.analyze_prosecutor_column(col, name, state_name=state_name)
                    if result:
                        result['location'] = state_name
                        result['is_notable'] = False
                        result['name'] = name
                        all_prosecutors.append(result)
                        
                        state_counts[state_name] = state_counts.get(state_name, 0) + 1
        
        print(f"✓ Processed {len([p for p in all_prosecutors if not p['is_notable']])} state prosecutors")
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
        
        return self
    
    def match_with_elections(self):
        """Match prosecutors with election data"""
        print_section("MATCHING WITH ELECTION DATA")
        
        def match_prosecutor(name):
            """Try to match a prosecutor name with election records"""
            name_parts = name.strip().split()
            if len(name_parts) < 2:
                return None
            
            first_name = name_parts[0]
            last_name = name_parts[-1]
            
            # Strategy 1: Exact first and last name match
            mask = (self.df_election['cand_fname'].str.lower() == first_name.lower()) & \
                   (self.df_election['cand_lname'].str.lower() == last_name.lower())
            matches = self.df_election[mask]
            
            # Strategy 2: Last name with first initial
            if len(matches) == 0:
                mask = self.df_election['cand_lname'].str.lower() == last_name.lower()
                matches = self.df_election[mask]
                if len(matches) > 0:
                    mask2 = matches['cand_fname'].str[0].str.lower() == first_name[0].lower()
                    if mask2.sum() > 0:
                        matches = matches[mask2]
            
            return matches if len(matches) > 0 else None
        
        # MATCH ALL PROSECUTORS (no rating threshold) for familiarity analyses
        print("Creating FULL matched sample (all prosecutors, any # of ratings)...")
        matched_data_all = []
        
        for idx, row in self.df_prosecutors.iterrows():
            matches = match_prosecutor(row['name'])
            
            if matches is not None:
                # Calculate electoral statistics
                election_stats = {
                    'num_elections': len(matches),
                    'years': sorted(matches['election_year'].unique().tolist()),
                    'ever_contested_primary': matches['primary_contested_reconciled'].eq('contested').any(),
                    'ever_contested_general': matches['general_contested_reconciled'].eq('contested').any(),
                    'ever_contested_any': False,  # Will calculate below
                    'ever_ran_as_incumbent': matches['incum_chall'].eq('I').any(),
                    'ever_ran_as_challenger': matches['incum_chall'].eq('C').any(),
                }
                
                # Get closest margins
                primary_margins = matches[matches['winner_primary'] == 'W']['vote_percent_primary'].dropna()
                general_margins = matches[matches['winner_general'] == 'W']['vote_percent_general'].dropna()
                
                election_stats['closest_primary_margin'] = primary_margins.min() if len(primary_margins) > 0 else np.nan
                election_stats['closest_general_margin'] = general_margins.min() if len(general_margins) > 0 else np.nan
                election_stats['had_close_primary'] = (primary_margins < Config.CLOSE_MARGIN_THRESHOLD).any() if len(primary_margins) > 0 else False
                election_stats['had_close_general'] = (general_margins < Config.CLOSE_MARGIN_THRESHOLD).any() if len(general_margins) > 0 else False
                
                # Calculate ever_contested_any
                election_stats['ever_contested_any'] = (
                    election_stats['ever_contested_primary'] or 
                    election_stats['ever_contested_general']
                )
                
                # Merge with prosecutor data
                result = {**row.to_dict(), **election_stats}
                matched_data_all.append(result)
        
        self.df_matched_all = pd.DataFrame(matched_data_all)
        
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
        
        if len(incumbents) > 0 and len(challengers) > 0:
            t_stat, p_val = ttest_ind(incumbents['familiarity_rate'], 
                                     challengers['familiarity_rate'])
            print(f"  T-test: t={t_stat:.3f}, p={p_val:.4f}")
            if p_val < 0.05:
                print("  *** SIGNIFICANT DIFFERENCE ***")
        
        self.results['incumbency_analysis'] = {
            'incumbent_mean': incumbents['familiarity_rate'].mean() if len(incumbents) > 0 else np.nan,
            'challenger_mean': challengers['familiarity_rate'].mean() if len(challengers) > 0 else np.nan
        }
        
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
        
        if len(contested) > 0 and len(uncontested) > 0:
            t_stat, p_val = ttest_ind(contested['familiarity_rate'], 
                                     uncontested['familiarity_rate'])
            print(f"  T-test: t={t_stat:.3f}, p={p_val:.4f}")
            if p_val < 0.05:
                print("  *** SIGNIFICANT DIFFERENCE ***")
        
        self.results['contestation_analysis'] = {
            'contested_mean': contested['familiarity_rate'].mean() if len(contested) > 0 else np.nan,
            'uncontested_mean': uncontested['familiarity_rate'].mean() if len(uncontested) > 0 else np.nan
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

        return self

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
        rq_path = self.run_research_questions()
        return summary_path, rq_path




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
    analyzer.analyze_contestation()
    analyzer.analyze_recall_risk()
    analyzer.run_multivariate_models()
    analyzer.export_results(Config.OUTPUT_DIR)

    # Extended integrated outputs
    reporter = IntegratedReporter(analyzer)
    summary_path, rq_path = reporter.export_all()
    print(f"\nExtended summary: {summary_path}")
    if rq_path:
        print(f"Research questions report: {rq_path}")

    print_section("✅ CORRECTED ANALYSIS COMPLETE!", "=")
    print(f"All outputs saved to: {Config.OUTPUT_DIR}/")
    print("\nGenerated files:")
    print("  📊 Data Files:")
    print("     - all_prosecutors_CORRECTED.csv")
    print("     - prosecutors_filtered_ideology_CORRECTED.csv")
    print("     - matched_prosecutors_FULL_CORRECTED.csv")
    print("     - matched_prosecutors_FILTERED_CORRECTED.csv")
    print("     - CORRECTED_FINDINGS_SUMMARY.txt")
    print("\n" + "="*80)
    print("\n🎯 NEXT STEPS:")
    print("1. Review CORRECTED_FINDINGS_SUMMARY.txt for key changes")
    print("2. Compare with old report to identify what needs updating")
    print("3. Pay special attention to state prosecutor familiarity rates!")
    print("="*80)


if __name__ == '__main__':
    main()

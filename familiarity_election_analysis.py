#!/usr/bin/env python3
"""
PROSECUTOR FAMILIARITY AND ELECTORAL COMPETITION ANALYSIS
===========================================================
This script analyzes the relationship between prosecutor familiarity
(as measured in the survey) and their electoral history, specifically:
1. Whether they participated in contested elections
2. Whether they participated in competitive/close elections

Author: Dvir Yogev, BERQ-J
Date: October 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import ttest_ind, mannwhitneyu, chi2_contingency
import statsmodels.api as sm
from pathlib import Path
import warnings
import re
warnings.filterwarnings('ignore')

# Set up plotting parameters
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = True
sns.set_style("whitegrid")


# =============================================================================
# UNDERSTANDING FAMILIARITY (from the original script)
# =============================================================================
"""
FAMILIARITY CONCEPT EXPLANATION:

In the original analysis script, "familiarity" is calculated as follows:

For each prosecutor in the survey:
1. Total_Ratings: Total number of responses (including "Not Familiar" responses 
   and blank/skipped responses)
2. Substantive_Ratings: Number of actual ideology ratings on the 1-4 scale:
   - Very Traditional (1)
   - Traditional (2)
   - Progressive (3)
   - Very Progressive (4)
3. Familiarity_Rate = (Substantive_Ratings / Total_Ratings) * 100

In other words, familiarity measures the percentage of respondents who actually 
rated a prosecutor's ideology (rather than saying "Not Familiar" or leaving it blank).

A prosecutor with:
- High familiarity (e.g., 80%+): Most people who encountered their name provided 
  an ideological rating
- Low familiarity (e.g., <20%): Most people either explicitly said "Not Familiar" 
  or left the rating blank

This analysis examines whether familiarity is related to electoral factors:
- Did more contested elections lead to higher name recognition?
- Did close/competitive races increase visibility?
"""


class FamiliarityElectionAnalyzer:
    """
    Analyzes the relationship between prosecutor familiarity and electoral competition
    """
    
    def __init__(self, survey_file, election_file):
        """
        Initialize the analyzer
        
        Parameters:
        -----------
        survey_file : str
            Path to the Qualtrics survey CSV file
        election_file : str
            Path to the election data CSV file
        """
        self.survey_file = survey_file
        self.election_file = election_file
        
        # Rating scale mapping
        self.rating_map = {
            'Very Traditional': 1,
            'Traditional': 2,
            'Not Familiar': np.nan,
            'Progressive': 3,
            'Very Progressive': 4
        }
        
        # Comprehensive DA name mapping for 50 national prosecutors
        self.da_names = {
            'notable_1': 'Larry Krasner|Philadelphia, PA',
            'notable_2': 'Alvin Bragg|Manhattan, NY',
            'notable_3': 'Mary Moriarty|Hennepin County, MN',
            'notable_4': 'Brooke Jenkins|San Francisco, CA',
            'notable_5': 'Chesa Boudin|San Francisco, CA',
            'notable_6': 'Pamela Price|Alameda County, CA',
            'notable_7': 'Nancy O\'Malley|Alameda County, CA',
            'notable_8': 'George Gascon|Los Angeles, CA',
            'notable_9': 'Nathan Hochman|Los Angeles, CA',
            'notable_10': 'Kim Foxx|Cook County (Chicago), IL',
            'notable_11': 'Eileen O\'Neill Burke|Cook County (Chicago), IL',
            'notable_12': 'Kim Gardner|St Louis, MO',
            'notable_13': 'Monique Worrell|Orlando, FL',
            'notable_14': 'Andrew Warren|Tampa, FL',
            'notable_15': 'Michael Dougherty|Boulder, CO',
            'notable_16': 'Sim Gill|Salt Lake City, UT',
            'notable_17': 'Eric Gonzalez|Brooklyn, NY',
            'notable_18': 'Eli Savit|Washtenaw County, MI',
            'notable_19': 'Kym Worthy|Wayne County (Detroit), MI',
            'notable_20': 'Summer Stephan|San Diego, CA',
            'notable_21': 'Katherine Fernandez-Rundle|Miami-Dade, FL',
            'notable_22': 'Melissa Nelson|Duval County (Jacksonville), FL',
            'notable_23': 'Mark Dupree|Wyandotte County (Kansas City), KS',
            'notable_24': 'Jose Garza|Travis County (Austin), TX',
            'notable_25': 'John Creuzot|Dallas County (Dallas), TX',
            'notable_26': 'Kim Ogg|Harris County (Houston), TX',
            'notable_27': 'Sean Teare|Harris County (Houston), TX',
            'notable_28': 'Joe Gonzales|Bexar County (San Antonio), TX',
            'notable_29': 'Rachael Rollins|Suffolk County (Boston), MA',
            'notable_30': 'Kevin Hayden|Suffolk County (Boston), MA',
            'notable_31': 'Ryan Mears|Marion County (Indianapolis), IN',
            'notable_32': 'Dan Satterberg|King County (Seattle), WA',
            'notable_33': 'Leesa Manion|King County (Seattle), WA',
            'notable_34': 'Jeff Rosen|Santa Clara County (San Jose), CA',
            'notable_35': 'Rachel Mitchell|Maricopa County (Phoenix), AZ',
            'notable_36': 'Laura Conover|Pima County (Tucson), AZ',
            'notable_37': 'Mike Schmidt|Multnomah County (Portland), OR',
            'notable_38': 'Nathan Vasquez|Multnomah County (Portland), OR',
            'notable_39': 'Amy Weirich|Shelby County (Memphis), TN',
            'notable_40': 'Steve Mulroy|Shelby County (Memphis), TN',
            'notable_41': 'Fani Willis|Fulton County (Atlanta), GA',
            'notable_42': 'Sherry Boston|DeKalb County (Atlanta), GA',
            'notable_43': 'Satana Deberry|Durham County, NC',
            'notable_44': 'Jason Williams|New Orleans, LA',
            'notable_45': 'Michael O\'Malley|Cuyahoga County (Cleveland), OH',
            'notable_46': 'Glenn Funk|Davidson County (Nashville), TN',
            'notable_47': 'Karl Racine|Washington, DC',
            'notable_48': 'Brian Schwalb|Washington, DC',
            'notable_49': 'Marilyn Mosby|Baltimore City, MD',
            'notable_50': 'Ivan Bates|Baltimore City, MD'
        }
        
        self.load_data()
    
    def load_data(self):
        """Load survey and election data"""
        print("=" * 80)
        print("LOADING DATA")
        print("=" * 80)
        
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
        
        # Load election data
        self.df_election = pd.read_csv(self.election_file)
        print(f"✓ Loaded {len(self.df_election)} election records")
        
        # Identify survey columns
        self.notable_cols = [col for col in self.df_survey.columns if col.startswith('notable_')]
        print(f"✓ Identified {len(self.notable_cols)} notable prosecutors")
        
        # Identify state DA columns
        self.state_da_cols = []
        for col in self.df_survey.columns:
            if '_' in col and not col.startswith('Q') and not col.startswith('notable'):
                if col not in ['gender_4_TEXT', 'race_1', 'race_2', 'race_3', 'race_4', 
                              'race_5', 'race_6', 'race_7', 'Q55_5_TEXT']:
                    unique_vals = self.df_survey[col].dropna().unique()
                    if len(unique_vals) > 0 and any('Progressive' in str(v) or 'Traditional' in str(v) 
                                                     for v in unique_vals):
                        self.state_da_cols.append(col)
        
        print(f"✓ Identified {len(self.state_da_cols)} state prosecutor columns")
        print()
    
    def calculate_familiarity_for_prosecutor(self, col):
        """
        Calculate familiarity rate for a specific prosecutor column
        Following the exact logic from the original script
        """
        ratings = self.df_survey[col].dropna()
        if len(ratings) == 0:
            return {
                'total_ratings': 0,
                'substantive_ratings': 0,
                'familiarity_rate': 0,
                'not_familiar': 0,
                'blank': 0
            }
        
        # Map to numeric values
        numeric = ratings.map(self.rating_map).dropna()
        not_familiar = (ratings == 'Not Familiar').sum()
        
        # In the actual survey data, blanks would already be NaN in the initial .dropna()
        # So we count those who responded at all
        total_ratings = len(ratings)
        substantive_ratings = len(numeric)
        
        return {
            'total_ratings': total_ratings,
            'substantive_ratings': substantive_ratings,
            'familiarity_rate': (substantive_ratings / total_ratings * 100) if total_ratings > 0 else 0,
            'not_familiar': not_familiar,
            'mean_score': numeric.mean() if len(numeric) > 0 else np.nan
        }
    
    def parse_prosecutor_name(self, full_name_location):
        """
        Parse prosecutor name from format: "FirstName LastName|Location"
        Returns first name and last name
        """
        if '|' in full_name_location:
            name_part = full_name_location.split('|')[0].strip()
        else:
            name_part = full_name_location.strip()
        
        # Split name - handle special cases
        parts = name_part.split()
        if len(parts) >= 2:
            first_name = parts[0]
            last_name = parts[-1]  # Take last part as last name
            return first_name, last_name
        return None, None
    
    def get_state_prosecutor_name(self, col):
        """Extract prosecutor name from state column using question text"""
        if self.questions_df is not None and col in self.questions_df.columns:
            question_text = self.questions_df[col].iloc[0]
            # Question format: "... - LastName\tFirstName\tCounty"
            if ' - ' in question_text:
                name_part = question_text.split(' - ')[-1].strip()
                # Split by tabs
                parts = name_part.split('\t')
                if len(parts) >= 2:
                    # Format is: LastName, FirstName, County
                    last_name = parts[0].strip()
                    first_name = parts[1].strip()
                    return first_name, last_name
        return None, None
    
    def match_with_elections(self, first_name, last_name):
        """
        Match prosecutor with election data and extract electoral history
        Returns dict with electoral metrics
        """
        # Clean names for matching
        first_clean = first_name.strip() if first_name else ''
        last_clean = last_name.strip() if last_name else ''
        
        # Find matches in election data
        matches = self.df_election[
            (self.df_election['cand_fname'].str.strip().str.lower() == first_clean.lower()) &
            (self.df_election['cand_lname'].str.strip().str.lower() == last_clean.lower())
        ].copy()
        
        if len(matches) == 0:
            return None
        
        # Get elections where they won (since survey only asks about winners)
        won_elections = matches[
            (matches['winner_primary'] == 'W') | (matches['winner_general'] == 'W')
        ].copy()
        
        if len(won_elections) == 0:
            return None
        
        # Calculate electoral metrics
        result = {
            'first_name': first_name,
            'last_name': last_name,
            'num_elections': len(won_elections['election_year'].unique()),
            'years': sorted(won_elections['election_year'].unique().tolist()),
            'ever_contested_primary': False,
            'ever_contested_general': False,
            'ever_contested_any': False,
            'num_contested_primaries': 0,
            'num_contested_generals': 0,
            'pct_contested_primaries': 0,
            'pct_contested_generals': 0,
            'ever_ran_as_challenger': False,
            'ever_ran_as_incumbent': False,
            'closest_primary_margin': np.nan,
            'closest_general_margin': np.nan,
            'had_close_primary': False,
            'had_close_general': False,
            'had_close_any': False
        }
        
        # Analyze each election
        for _, row in won_elections.iterrows():
            # Check contested status
            if row['primary_contested_reconciled'] == 'contested':
                result['ever_contested_primary'] = True
                result['num_contested_primaries'] += 1
            
            if row['general_contested_reconciled'] == 'contested':
                result['ever_contested_general'] = True
                result['num_contested_generals'] += 1
            
            # Check incumbency status
            if pd.notna(row['incum_chall']):
                if row['incum_chall'] == 'I':
                    result['ever_ran_as_incumbent'] = True
                elif row['incum_chall'] == 'C':
                    result['ever_ran_as_challenger'] = True
            
            # Analyze competitiveness for contested races
            year = row['election_year']
            state = row['state']
            district = row['district']
            
            # PRIMARY COMPETITIVENESS
            if row['primary_contested_reconciled'] == 'contested' and pd.notna(row['vote_percent_primary']):
                # Get all primary candidates in this race
                primary_race = self.df_election[
                    (self.df_election['election_year'] == year) &
                    (self.df_election['state'] == state) &
                    (self.df_election['district'] == district) &
                    (self.df_election['ran_primary'] == 'Y') &
                    (self.df_election['vote_percent_primary'].notna())
                ].copy()
                
                if len(primary_race) > 1:
                    # Sort by vote percentage
                    primary_race = primary_race.sort_values('vote_percent_primary', ascending=False)
                    winner_pct = primary_race.iloc[0]['vote_percent_primary']
                    runnerup_pct = primary_race.iloc[1]['vote_percent_primary']
                    margin = winner_pct - runnerup_pct
                    
                    # Update closest margin
                    if pd.isna(result['closest_primary_margin']) or margin < result['closest_primary_margin']:
                        result['closest_primary_margin'] = margin
            
            # GENERAL COMPETITIVENESS
            if row['general_contested_reconciled'] == 'contested' and pd.notna(row['vote_percent_general']):
                # Get all general election candidates in this race
                general_race = self.df_election[
                    (self.df_election['election_year'] == year) &
                    (self.df_election['state'] == state) &
                    (self.df_election['district'] == district) &
                    (self.df_election['ran_general'] == 'Y') &
                    (self.df_election['vote_percent_general'].notna())
                ].copy()
                
                if len(general_race) > 1:
                    # Sort by vote percentage
                    general_race = general_race.sort_values('vote_percent_general', ascending=False)
                    winner_pct = general_race.iloc[0]['vote_percent_general']
                    runnerup_pct = general_race.iloc[1]['vote_percent_general']
                    margin = winner_pct - runnerup_pct
                    
                    # Update closest margin
                    if pd.isna(result['closest_general_margin']) or margin < result['closest_general_margin']:
                        result['closest_general_margin'] = margin
        
        # Calculate aggregate metrics
        result['ever_contested_any'] = result['ever_contested_primary'] or result['ever_contested_general']
        
        if result['num_elections'] > 0:
            result['pct_contested_primaries'] = (result['num_contested_primaries'] / result['num_elections'] * 100)
            result['pct_contested_generals'] = (result['num_contested_generals'] / result['num_elections'] * 100)
        
        # Define "close" as margin < 15 percentage points (adjustable threshold)
        CLOSE_MARGIN_THRESHOLD = 15
        if pd.notna(result['closest_primary_margin']):
            result['had_close_primary'] = result['closest_primary_margin'] < CLOSE_MARGIN_THRESHOLD
        
        if pd.notna(result['closest_general_margin']):
            result['had_close_general'] = result['closest_general_margin'] < CLOSE_MARGIN_THRESHOLD
        
        result['had_close_any'] = result['had_close_primary'] or result['had_close_general']
        
        return result
    
    def analyze_notable_prosecutors(self):
        """
        Analyze the 50 notable prosecutors: familiarity vs electoral competition
        """
        print("=" * 80)
        print("PART 1: NOTABLE PROSECUTORS (National List of 50)")
        print("=" * 80)
        print()
        
        results = []
        
        for col in self.notable_cols:
            # Get prosecutor info
            if col in self.da_names:
                full_name = self.da_names[col]
                name_part, location = full_name.split('|')
                first_name, last_name = self.parse_prosecutor_name(full_name)
            else:
                continue
            
            # Calculate familiarity
            fam_stats = self.calculate_familiarity_for_prosecutor(col)
            
            # Match with election data
            election_data = self.match_with_elections(first_name, last_name)
            
            # Combine results
            result = {
                'column': col,
                'name': name_part.strip(),
                'location': location.strip(),
                'first_name': first_name,
                'last_name': last_name,
                **fam_stats
            }
            
            if election_data:
                result.update(election_data)
            else:
                # No election match - set defaults
                result.update({
                    'num_elections': 0,
                    'years': [],
                    'ever_contested_primary': False,
                    'ever_contested_general': False,
                    'ever_contested_any': False,
                    'num_contested_primaries': 0,
                    'num_contested_generals': 0,
                    'pct_contested_primaries': 0,
                    'pct_contested_generals': 0,
                    'ever_ran_as_challenger': False,
                    'ever_ran_as_incumbent': False,
                    'closest_primary_margin': np.nan,
                    'closest_general_margin': np.nan,
                    'had_close_primary': False,
                    'had_close_general': False,
                    'had_close_any': False
                })
            
            results.append(result)
        
        self.notable_df = pd.DataFrame(results)
        
        # Print summary
        matched = self.notable_df['num_elections'] > 0
        print(f"Total notable prosecutors: {len(self.notable_df)}")
        print(f"Matched with election data: {matched.sum()} ({matched.sum()/len(self.notable_df)*100:.1f}%)")
        print(f"Not matched: {(~matched).sum()}")
        print()
        
        return self.notable_df
    
    def analyze_state_prosecutors(self):
        """
        Analyze state-level prosecutors: familiarity vs electoral competition
        """
        print("=" * 80)
        print("PART 2: STATE PROSECUTORS (Within-State Lists)")
        print("=" * 80)
        print()
        
        results = []
        
        for col in self.state_da_cols:
            # Extract state from column name
            state_name = col.rsplit('_', 1)[0]
            
            # Get prosecutor name from question text
            first_name, last_name = self.get_state_prosecutor_name(col)
            
            if not first_name or not last_name:
                continue
            
            # Calculate familiarity (only for those who agreed to rate state DAs)
            if 'more' in self.df_survey.columns:
                agreed_to_rate = self.df_survey['more'] == 'Show me the prosecutors from my state'
                respondent_state = self.df_survey.get('Q1', None)
                
                # Filter for this specific state
                if respondent_state is not None:
                    state_respondents = agreed_to_rate & (self.df_survey['Q1'] == state_name)
                    subset = self.df_survey[state_respondents]
                    
                    if len(subset) == 0:
                        continue
                    
                    ratings = subset[col].dropna()
                    if len(ratings) == 0:
                        continue
                    
                    numeric = ratings.map(self.rating_map).dropna()
                    not_familiar = (ratings == 'Not Familiar').sum()
                    
                    fam_stats = {
                        'total_ratings': len(ratings),
                        'substantive_ratings': len(numeric),
                        'familiarity_rate': (len(numeric) / len(ratings) * 100) if len(ratings) > 0 else 0,
                        'not_familiar': not_familiar,
                        'mean_score': numeric.mean() if len(numeric) > 0 else np.nan
                    }
                else:
                    continue
            else:
                continue
            
            # Match with election data
            election_data = self.match_with_elections(first_name, last_name)
            
            # Combine results
            result = {
                'column': col,
                'state': state_name,
                'name': f"{first_name} {last_name}",
                'first_name': first_name,
                'last_name': last_name,
                **fam_stats
            }
            
            if election_data:
                result.update(election_data)
            else:
                result.update({
                    'num_elections': 0,
                    'years': [],
                    'ever_contested_primary': False,
                    'ever_contested_general': False,
                    'ever_contested_any': False,
                    'num_contested_primaries': 0,
                    'num_contested_generals': 0,
                    'pct_contested_primaries': 0,
                    'pct_contested_generals': 0,
                    'ever_ran_as_challenger': False,
                    'ever_ran_as_incumbent': False,
                    'closest_primary_margin': np.nan,
                    'closest_general_margin': np.nan,
                    'had_close_primary': False,
                    'had_close_general': False,
                    'had_close_any': False
                })
            
            results.append(result)
        
        self.state_df = pd.DataFrame(results)
        
        if len(self.state_df) > 0:
            matched = self.state_df['num_elections'] > 0
            print(f"Total state prosecutors analyzed: {len(self.state_df)}")
            print(f"Matched with election data: {matched.sum()} ({matched.sum()/len(self.state_df)*100:.1f}%)")
            print(f"Not matched: {(~matched).sum()}")
        else:
            print("No state prosecutors found in data")
        print()
        
        return self.state_df
    
    def analyze_contested_elections_relationship(self, df, title):
        """
        Analyze relationship between familiarity and contested elections
        """
        print("=" * 80)
        print(f"ANALYSIS: {title}")
        print("=" * 80)
        print()
        
        # Filter to those with election data
        df_with_elections = df[df['num_elections'] > 0].copy()
        
        if len(df_with_elections) == 0:
            print("No prosecutors with election data found.")
            print()
            return None
        
        print(f"Prosecutors with election data: {len(df_with_elections)}")
        print()
        
        # CONTESTED VS UNCONTESTED
        print("--- CONTESTED ELECTIONS (ANY) ---")
        contested_any = df_with_elections[df_with_elections['ever_contested_any']]
        uncontested = df_with_elections[~df_with_elections['ever_contested_any']]
        
        print(f"Ever contested (primary or general): {len(contested_any)}")
        print(f"Never contested: {len(uncontested)}")
        
        if len(contested_any) > 0 and len(uncontested) > 0:
            print(f"\nFamiliarity rates:")
            print(f"  Contested: Mean = {contested_any['familiarity_rate'].mean():.2f}%, "
                  f"Median = {contested_any['familiarity_rate'].median():.2f}%")
            print(f"  Uncontested: Mean = {uncontested['familiarity_rate'].mean():.2f}%, "
                  f"Median = {uncontested['familiarity_rate'].median():.2f}%")
            print(f"  Difference: {contested_any['familiarity_rate'].mean() - uncontested['familiarity_rate'].mean():+.2f}%")
            
            # Statistical test
            t_stat, p_val = ttest_ind(contested_any['familiarity_rate'], 
                                       uncontested['familiarity_rate'], 
                                       nan_policy='omit')
            u_stat, p_val_mw = mannwhitneyu(contested_any['familiarity_rate'], 
                                            uncontested['familiarity_rate'])
            print(f"\n  T-test: t={t_stat:.3f}, p={p_val:.4f}")
            print(f"  Mann-Whitney U: U={u_stat:.3f}, p={p_val_mw:.4f}")
        print()
        
        # PRIMARY CONTESTED
        print("--- CONTESTED PRIMARY ELECTIONS ---")
        contested_prim = df_with_elections[df_with_elections['ever_contested_primary']]
        uncontested_prim = df_with_elections[~df_with_elections['ever_contested_primary']]
        
        print(f"Ever contested primary: {len(contested_prim)}")
        print(f"Never contested primary: {len(uncontested_prim)}")
        
        if len(contested_prim) > 0 and len(uncontested_prim) > 0:
            print(f"\nFamiliarity rates:")
            print(f"  Contested primary: Mean = {contested_prim['familiarity_rate'].mean():.2f}%, "
                  f"Median = {contested_prim['familiarity_rate'].median():.2f}%")
            print(f"  No contested primary: Mean = {uncontested_prim['familiarity_rate'].mean():.2f}%, "
                  f"Median = {uncontested_prim['familiarity_rate'].median():.2f}%")
            print(f"  Difference: {contested_prim['familiarity_rate'].mean() - uncontested_prim['familiarity_rate'].mean():+.2f}%")
            
            t_stat, p_val = ttest_ind(contested_prim['familiarity_rate'], 
                                       uncontested_prim['familiarity_rate'], 
                                       nan_policy='omit')
            u_stat, p_val_mw = mannwhitneyu(contested_prim['familiarity_rate'], 
                                            uncontested_prim['familiarity_rate'])
            print(f"\n  T-test: t={t_stat:.3f}, p={p_val:.4f}")
            print(f"  Mann-Whitney U: U={u_stat:.3f}, p={p_val_mw:.4f}")
        print()
        
        # GENERAL CONTESTED
        print("--- CONTESTED GENERAL ELECTIONS ---")
        contested_gen = df_with_elections[df_with_elections['ever_contested_general']]
        uncontested_gen = df_with_elections[~df_with_elections['ever_contested_general']]
        
        print(f"Ever contested general: {len(contested_gen)}")
        print(f"Never contested general: {len(uncontested_gen)}")
        
        if len(contested_gen) > 0 and len(uncontested_gen) > 0:
            print(f"\nFamiliarity rates:")
            print(f"  Contested general: Mean = {contested_gen['familiarity_rate'].mean():.2f}%, "
                  f"Median = {contested_gen['familiarity_rate'].median():.2f}%")
            print(f"  No contested general: Mean = {uncontested_gen['familiarity_rate'].mean():.2f}%, "
                  f"Median = {uncontested_gen['familiarity_rate'].median():.2f}%")
            print(f"  Difference: {contested_gen['familiarity_rate'].mean() - uncontested_gen['familiarity_rate'].mean():+.2f}%")
            
            t_stat, p_val = ttest_ind(contested_gen['familiarity_rate'], 
                                       uncontested_gen['familiarity_rate'], 
                                       nan_policy='omit')
            u_stat, p_val_mw = mannwhitneyu(contested_gen['familiarity_rate'], 
                                            uncontested_gen['familiarity_rate'])
            print(f"\n  T-test: t={t_stat:.3f}, p={p_val:.4f}")
            print(f"  Mann-Whitney U: U={u_stat:.3f}, p={p_val_mw:.4f}")
        print()
        
        return df_with_elections
    
    def analyze_close_elections_relationship(self, df, title):
        """
        Analyze relationship between familiarity and close/competitive elections
        """
        print("=" * 80)
        print(f"ANALYSIS: {title}")
        print("=" * 80)
        print()
        
        # Filter to those with election data
        df_with_elections = df[df['num_elections'] > 0].copy()
        
        if len(df_with_elections) == 0:
            print("No prosecutors with election data found.")
            print()
            return None
        
        # CLOSE ELECTIONS (ANY)
        print("--- CLOSE/COMPETITIVE ELECTIONS (MARGIN < 15%) ---")
        had_close = df_with_elections[df_with_elections['had_close_any']]
        no_close = df_with_elections[~df_with_elections['had_close_any']]
        
        print(f"Had close election (primary or general): {len(had_close)}")
        print(f"No close elections: {len(no_close)}")
        
        if len(had_close) > 0 and len(no_close) > 0:
            print(f"\nFamiliarity rates:")
            print(f"  Had close election: Mean = {had_close['familiarity_rate'].mean():.2f}%, "
                  f"Median = {had_close['familiarity_rate'].median():.2f}%")
            print(f"  No close elections: Mean = {no_close['familiarity_rate'].mean():.2f}%, "
                  f"Median = {no_close['familiarity_rate'].median():.2f}%")
            print(f"  Difference: {had_close['familiarity_rate'].mean() - no_close['familiarity_rate'].mean():+.2f}%")
            
            t_stat, p_val = ttest_ind(had_close['familiarity_rate'], 
                                       no_close['familiarity_rate'], 
                                       nan_policy='omit')
            u_stat, p_val_mw = mannwhitneyu(had_close['familiarity_rate'], 
                                            no_close['familiarity_rate'])
            print(f"\n  T-test: t={t_stat:.3f}, p={p_val:.4f}")
            print(f"  Mann-Whitney U: U={u_stat:.3f}, p={p_val_mw:.4f}")
        print()
        
        # CLOSE PRIMARY
        print("--- CLOSE PRIMARY ELECTIONS ---")
        close_prim = df_with_elections[df_with_elections['had_close_primary']]
        not_close_prim = df_with_elections[~df_with_elections['had_close_primary'] & 
                                          df_with_elections['ever_contested_primary']]
        
        print(f"Had close primary (margin < 15%): {len(close_prim)}")
        print(f"Had contested but not close primary: {len(not_close_prim)}")
        
        if len(close_prim) > 0 and len(not_close_prim) > 0:
            print(f"\nFamiliarity rates:")
            print(f"  Close primary: Mean = {close_prim['familiarity_rate'].mean():.2f}%, "
                  f"Median = {close_prim['familiarity_rate'].median():.2f}%")
            print(f"  Not close primary: Mean = {not_close_prim['familiarity_rate'].mean():.2f}%, "
                  f"Median = {not_close_prim['familiarity_rate'].median():.2f}%")
            print(f"  Difference: {close_prim['familiarity_rate'].mean() - not_close_prim['familiarity_rate'].mean():+.2f}%")
            
            t_stat, p_val = ttest_ind(close_prim['familiarity_rate'], 
                                       not_close_prim['familiarity_rate'], 
                                       nan_policy='omit')
            print(f"\n  T-test: t={t_stat:.3f}, p={p_val:.4f}")
        print()
        
        # CLOSE GENERAL
        print("--- CLOSE GENERAL ELECTIONS ---")
        close_gen = df_with_elections[df_with_elections['had_close_general']]
        not_close_gen = df_with_elections[~df_with_elections['had_close_general'] & 
                                         df_with_elections['ever_contested_general']]
        
        print(f"Had close general (margin < 15%): {len(close_gen)}")
        print(f"Had contested but not close general: {len(not_close_gen)}")
        
        if len(close_gen) > 0 and len(not_close_gen) > 0:
            print(f"\nFamiliarity rates:")
            print(f"  Close general: Mean = {close_gen['familiarity_rate'].mean():.2f}%, "
                  f"Median = {close_gen['familiarity_rate'].median():.2f}%")
            print(f"  Not close general: Mean = {not_close_gen['familiarity_rate'].mean():.2f}%, "
                  f"Median = {not_close_gen['familiarity_rate'].median():.2f}%")
            print(f"  Difference: {close_gen['familiarity_rate'].mean() - not_close_gen['familiarity_rate'].mean():+.2f}%")
            
            t_stat, p_val = ttest_ind(close_gen['familiarity_rate'], 
                                       not_close_gen['familiarity_rate'], 
                                       nan_policy='omit')
            print(f"\n  T-test: t={t_stat:.3f}, p={p_val:.4f}")
        print()
        
        # MARGIN CORRELATION
        print("--- CORRELATION: FAMILIARITY vs ELECTION MARGIN ---")
        
        # Primary margin correlation
        has_prim_margin = df_with_elections['closest_primary_margin'].notna()
        if has_prim_margin.sum() > 2:
            corr, p_val = stats.pearsonr(
                df_with_elections.loc[has_prim_margin, 'familiarity_rate'],
                df_with_elections.loc[has_prim_margin, 'closest_primary_margin']
            )
            print(f"Primary margin: r={corr:.3f}, p={p_val:.4f} (n={has_prim_margin.sum()})")
            print(f"  Interpretation: {'Negative' if corr < 0 else 'Positive'} correlation - "
                  f"{'closer races = higher familiarity' if corr < 0 else 'closer races = lower familiarity'}")
        
        # General margin correlation
        has_gen_margin = df_with_elections['closest_general_margin'].notna()
        if has_gen_margin.sum() > 2:
            corr, p_val = stats.pearsonr(
                df_with_elections.loc[has_gen_margin, 'familiarity_rate'],
                df_with_elections.loc[has_gen_margin, 'closest_general_margin']
            )
            print(f"General margin: r={corr:.3f}, p={p_val:.4f} (n={has_gen_margin.sum()})")
            print(f"  Interpretation: {'Negative' if corr < 0 else 'Positive'} correlation - "
                  f"{'closer races = higher familiarity' if corr < 0 else 'closer races = lower familiarity'}")
        
        print()
        
        return df_with_elections
    
    def create_visualizations(self, output_dir='output'):
        """Create visualizations of the relationships"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print("=" * 80)
        print("CREATING VISUALIZATIONS")
        print("=" * 80)
        print()
        
        # VISUALIZATION 1: Contested vs Uncontested (Notable)
        if hasattr(self, 'notable_df') and len(self.notable_df) > 0:
            df_plot = self.notable_df[self.notable_df['num_elections'] > 0].copy()
            
            if len(df_plot) > 0:
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                
                # Panel 1: Any contested
                ax = axes[0]
                contested_data = [
                    df_plot[df_plot['ever_contested_any']]['familiarity_rate'],
                    df_plot[~df_plot['ever_contested_any']]['familiarity_rate']
                ]
                ax.boxplot(contested_data, labels=['Ever Contested', 'Never Contested'])
                ax.set_ylabel('Familiarity Rate (%)', fontsize=11)
                ax.set_title('Contested Elections (Any)\nNotable Prosecutors', fontsize=12, fontweight='bold')
                ax.grid(axis='y', alpha=0.3)
                
                # Panel 2: Close elections
                ax = axes[1]
                close_data = [
                    df_plot[df_plot['had_close_any']]['familiarity_rate'],
                    df_plot[~df_plot['had_close_any']]['familiarity_rate']
                ]
                ax.boxplot(close_data, labels=['Had Close Race', 'No Close Race'])
                ax.set_ylabel('Familiarity Rate (%)', fontsize=11)
                ax.set_title('Close Elections (Margin < 15%)\nNotable Prosecutors', fontsize=12, fontweight='bold')
                ax.grid(axis='y', alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(output_path / 'familiarity_elections_notable.png', dpi=300, bbox_inches='tight')
                plt.close()
                print("✓ Created: familiarity_elections_notable.png")
        
        # VISUALIZATION 2: State prosecutors
        if hasattr(self, 'state_df') and len(self.state_df) > 0:
            df_plot = self.state_df[self.state_df['num_elections'] > 0].copy()
            
            if len(df_plot) > 5:  # Need enough data for meaningful plots
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                
                # Panel 1: Any contested
                ax = axes[0]
                contested_data = [
                    df_plot[df_plot['ever_contested_any']]['familiarity_rate'],
                    df_plot[~df_plot['ever_contested_any']]['familiarity_rate']
                ]
                ax.boxplot(contested_data, labels=['Ever Contested', 'Never Contested'])
                ax.set_ylabel('Familiarity Rate (%)', fontsize=11)
                ax.set_title('Contested Elections (Any)\nState Prosecutors', fontsize=12, fontweight='bold')
                ax.grid(axis='y', alpha=0.3)
                
                # Panel 2: Close elections
                ax = axes[1]
                close_data = [
                    df_plot[df_plot['had_close_any']]['familiarity_rate'],
                    df_plot[~df_plot['had_close_any']]['familiarity_rate']
                ]
                ax.boxplot(close_data, labels=['Had Close Race', 'No Close Race'])
                ax.set_ylabel('Familiarity Rate (%)', fontsize=11)
                ax.set_title('Close Elections (Margin < 15%)\nState Prosecutors', fontsize=12, fontweight='bold')
                ax.grid(axis='y', alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(output_path / 'familiarity_elections_state.png', dpi=300, bbox_inches='tight')
                plt.close()
                print("✓ Created: familiarity_elections_state.png")
        
        # VISUALIZATION 3: Scatter plot - margin vs familiarity
        if hasattr(self, 'notable_df'):
            df_plot = self.notable_df[self.notable_df['num_elections'] > 0].copy()
            
            # Create figure with two panels
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Primary margin
            ax = axes[0]
            has_data = df_plot['closest_primary_margin'].notna()
            if has_data.sum() > 2:
                ax.scatter(df_plot.loc[has_data, 'closest_primary_margin'],
                          df_plot.loc[has_data, 'familiarity_rate'],
                          alpha=0.6, s=80, color='steelblue')
                
                # Add trend line
                mask = has_data
                x = df_plot.loc[mask, 'closest_primary_margin']
                y = df_plot.loc[mask, 'familiarity_rate']
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                ax.plot(x, p(x), "r--", alpha=0.8, linewidth=2)
                
                ax.set_xlabel('Closest Primary Margin (%)', fontsize=11)
                ax.set_ylabel('Familiarity Rate (%)', fontsize=11)
                ax.set_title('Primary Election Competitiveness\nvs Familiarity', 
                           fontsize=12, fontweight='bold')
                ax.grid(alpha=0.3)
            
            # General margin
            ax = axes[1]
            has_data = df_plot['closest_general_margin'].notna()
            if has_data.sum() > 2:
                ax.scatter(df_plot.loc[has_data, 'closest_general_margin'],
                          df_plot.loc[has_data, 'familiarity_rate'],
                          alpha=0.6, s=80, color='darkgreen')
                
                # Add trend line
                mask = has_data
                x = df_plot.loc[mask, 'closest_general_margin']
                y = df_plot.loc[mask, 'familiarity_rate']
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                ax.plot(x, p(x), "r--", alpha=0.8, linewidth=2)
                
                ax.set_xlabel('Closest General Margin (%)', fontsize=11)
                ax.set_ylabel('Familiarity Rate (%)', fontsize=11)
                ax.set_title('General Election Competitiveness\nvs Familiarity', 
                           fontsize=12, fontweight='bold')
                ax.grid(alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_path / 'margin_familiarity_scatter.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ Created: margin_familiarity_scatter.png")
        
        print()
    
    def export_results(self, output_dir='output'):
        """Export results to CSV files"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print("=" * 80)
        print("EXPORTING RESULTS")
        print("=" * 80)
        print()
        
        if hasattr(self, 'notable_df'):
            # Convert list columns to strings for CSV export
            notable_export = self.notable_df.copy()
            notable_export['years'] = notable_export['years'].apply(lambda x: str(x) if isinstance(x, list) else '')
            notable_export.to_csv(output_path / 'notable_prosecutors_elections.csv', index=False)
            print("✓ Exported: notable_prosecutors_elections.csv")
        
        if hasattr(self, 'state_df') and len(self.state_df) > 0:
            state_export = self.state_df.copy()
            state_export['years'] = state_export['years'].apply(lambda x: str(x) if isinstance(x, list) else '')
            state_export.to_csv(output_path / 'state_prosecutors_elections.csv', index=False)
            print("✓ Exported: state_prosecutors_elections.csv")
        
        print()
    
    def generate_summary_report(self, output_dir='output'):
        """Generate a comprehensive summary report"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        report = """
================================================================================
PROSECUTOR FAMILIARITY AND ELECTORAL COMPETITION ANALYSIS
================================================================================

UNDERSTANDING FAMILIARITY:
--------------------------
Familiarity is calculated as: (Substantive Ratings / Total Ratings) × 100

Where:
- Substantive Ratings: Number of respondents who rated the prosecutor on the
  1-4 ideological scale (Very Traditional to Very Progressive)
- Total Ratings: All responses, including "Not Familiar" and blanks

High familiarity = Most people provided an ideological rating
Low familiarity = Most people said "Not Familiar" or left it blank

================================================================================
KEY FINDINGS
================================================================================

"""
        
        # Add notable prosecutors summary
        if hasattr(self, 'notable_df'):
            df = self.notable_df[self.notable_df['num_elections'] > 0]
            
            if len(df) > 0:
                report += f"""
NOTABLE PROSECUTORS (National List of 50):
------------------------------------------
Total prosecutors analyzed: {len(self.notable_df)}
Matched with election data: {len(df)} ({len(df)/len(self.notable_df)*100:.1f}%)

CONTESTED ELECTIONS:
"""
                contested = df[df['ever_contested_any']]
                uncontested = df[~df['ever_contested_any']]
                
                if len(contested) > 0 and len(uncontested) > 0:
                    report += f"""  Ever contested: {len(contested)} prosecutors
  Never contested: {len(uncontested)} prosecutors
  
  Mean familiarity:
    - Contested: {contested['familiarity_rate'].mean():.2f}%
    - Uncontested: {uncontested['familiarity_rate'].mean():.2f}%
    - Difference: {contested['familiarity_rate'].mean() - uncontested['familiarity_rate'].mean():+.2f}%
"""
                
                report += f"""
CLOSE ELECTIONS (margin < 15%):
"""
                close = df[df['had_close_any']]
                not_close = df[~df['had_close_any']]
                
                if len(close) > 0 and len(not_close) > 0:
                    report += f"""  Had close race: {len(close)} prosecutors
  No close race: {len(not_close)} prosecutors
  
  Mean familiarity:
    - Close race: {close['familiarity_rate'].mean():.2f}%
    - No close race: {not_close['familiarity_rate'].mean():.2f}%
    - Difference: {close['familiarity_rate'].mean() - not_close['familiarity_rate'].mean():+.2f}%
"""
        
        # Add state prosecutors summary
        if hasattr(self, 'state_df') and len(self.state_df) > 0:
            df = self.state_df[self.state_df['num_elections'] > 0]
            
            if len(df) > 0:
                report += f"""
================================================================================
STATE PROSECUTORS (Within-State Lists):
---------------------------------------
Total prosecutors analyzed: {len(self.state_df)}
Matched with election data: {len(df)} ({len(df)/len(self.state_df)*100:.1f}%)

CONTESTED ELECTIONS:
"""
                contested = df[df['ever_contested_any']]
                uncontested = df[~df['ever_contested_any']]
                
                if len(contested) > 0 and len(uncontested) > 0:
                    report += f"""  Ever contested: {len(contested)} prosecutors
  Never contested: {len(uncontested)} prosecutors
  
  Mean familiarity:
    - Contested: {contested['familiarity_rate'].mean():.2f}%
    - Uncontested: {uncontested['familiarity_rate'].mean():.2f}%
    - Difference: {contested['familiarity_rate'].mean() - uncontested['familiarity_rate'].mean():+.2f}%
"""
        
        report += """
================================================================================
INTERPRETATION GUIDE
================================================================================

1. CONTESTED vs UNCONTESTED:
   - Positive difference: Prosecutors who faced electoral competition have
     higher name recognition among criminal justice professionals
   - Negative difference: Uncontested prosecutors are better known
   - No difference: Electoral competition doesn't affect familiarity

2. CLOSE vs NON-CLOSE ELECTIONS:
   - Positive difference: Competitive races (margin < 15%) increase visibility
   - Negative difference: Landslide victories lead to higher recognition
   - No difference: Margin of victory doesn't affect familiarity

3. CORRELATION WITH MARGIN:
   - Negative correlation: Closer races = higher familiarity
   - Positive correlation: Larger margins = higher familiarity
   - No correlation: Competitiveness doesn't relate to familiarity

================================================================================
METHODOLOGICAL NOTES
================================================================================

1. Survey only asked about prosecutors who won elections and were in office
2. Election data covers candidate-county-year observations
3. "Close" elections defined as margin < 15 percentage points
4. Multi-candidate races properly account for runner-up position
5. Analysis includes both primary and general elections
6. Statistical tests used: t-tests, Mann-Whitney U, Pearson correlation

================================================================================
"""
        
        # Save report
        with open(output_path / 'familiarity_elections_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(report)
        print("\n✅ Report saved to: familiarity_elections_report.txt")
        print()
    
    def run_complete_analysis(self, output_dir='output'):
        """Run the complete analysis pipeline"""
        print("\n" + "=" * 80)
        print("FAMILIARITY AND ELECTORAL COMPETITION ANALYSIS")
        print("=" * 80)
        print()
        
        # Analyze notable prosecutors
        self.analyze_notable_prosecutors()
        
        # Analyze state prosecutors
        self.analyze_state_prosecutors()
        
        # Relationship analyses
        if hasattr(self, 'notable_df'):
            self.analyze_contested_elections_relationship(
                self.notable_df, 
                "Notable Prosecutors - Contested Elections"
            )
            self.analyze_close_elections_relationship(
                self.notable_df,
                "Notable Prosecutors - Close Elections"
            )
        
        if hasattr(self, 'state_df') and len(self.state_df) > 0:
            self.analyze_contested_elections_relationship(
                self.state_df,
                "State Prosecutors - Contested Elections"
            )
            self.analyze_close_elections_relationship(
                self.state_df,
                "State Prosecutors - Close Elections"
            )
        
        # Create visualizations
        self.create_visualizations(output_dir)
        
        # Export results
        self.export_results(output_dir)
        
        # Generate report
        self.generate_summary_report(output_dir)
        
        print("=" * 80)
        print("✅ ANALYSIS COMPLETE!")
        print("=" * 80)
        print(f"\nAll outputs saved to: {output_dir}/")
        print("\nGenerated files:")
        print("  📊 Data:")
        print("     - notable_prosecutors_elections.csv")
        print("     - state_prosecutors_elections.csv")
        print("\n  📈 Visualizations:")
        print("     - familiarity_elections_notable.png")
        print("     - familiarity_elections_state.png")
        print("     - margin_familiarity_scatter.png")
        print("\n  📄 Report:")
        print("     - familiarity_elections_report.txt")
        print("\n" + "=" * 80)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function"""
    
    # File paths
    survey_file = 'PP_survey_draft_October_16__2025_13_32.csv'
    election_file = 'elections_with_reconciled_contested.csv'
    output_dir = 'output'
    
    print("\n" + "=" * 80)
    print("PROSECUTOR FAMILIARITY AND ELECTORAL COMPETITION ANALYSIS")
    print("=" * 80)
    print(f"\n📁 Survey file: {survey_file}")
    print(f"📁 Election file: {election_file}")
    print(f"📁 Output directory: {output_dir}")
    print("=" * 80)
    
    # Initialize analyzer
    analyzer = FamiliarityElectionAnalyzer(
        survey_file=survey_file,
        election_file=election_file
    )
    
    # Run complete analysis
    analyzer.run_complete_analysis(output_dir=output_dir)


if __name__ == '__main__':
    main()

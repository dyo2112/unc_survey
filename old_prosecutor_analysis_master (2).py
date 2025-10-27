#!/usr/bin/env python3
"""
================================================================================
PROSECUTOR IDEOLOGY SURVEY - COMPREHENSIVE ANALYSIS
================================================================================

This is the MASTER SCRIPT that performs ALL analyses from start to finish:
- Data loading and cleaning
- Survey analysis (familiarity, ideology ratings)
- Electoral data matching
- Interaction models
- Predictive models
- Comprehensive visualizations
- Report generation

Author: Dvir Yogev, BERQ-J
Institution: Criminal Law & Justice Center, UC Berkeley Law
Date: October 2025
Version: 2.0 - COMPREHENSIVE EDITION

USAGE:
    python prosecutor_analysis_master.py
    
OUTPUT:
    - All CSVs in 'output/' directory
    - All visualizations in 'output/visualizations/'
    - Comprehensive report in 'output/FINAL_REPORT.txt'
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
    OUTPUT_DIR = 'output'
    VIZ_DIR = 'output/visualizations'
    
    # Analysis parameters
    MIN_RATINGS_THRESHOLD = 10  # Minimum ratings for reliable analysis
    CLOSE_MARGIN_THRESHOLD = 55  # % threshold for "close" elections
    
    # Rating scale
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
    """Load and prepare survey and election data"""
    
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
                if col not in ['gender_4_TEXT', 'race_1', 'race_2', 'race_3', 'race_4', 
                              'race_5', 'race_6', 'race_7', 'Q55_5_TEXT']:
                    unique_vals = self.df_survey[col].dropna().unique()
                    if len(unique_vals) > 0 and any('Progressive' in str(v) or 'Traditional' in str(v) 
                                                     for v in unique_vals):
                        self.state_da_cols.append(col)
        
        print(f"✓ Identified {len(self.state_da_cols)} state prosecutor columns")
        
        return self


# ================================================================================
# PROSECUTOR ANALYZER CLASS
# ================================================================================

class ProsecutorAnalyzer:
    """Main analyzer class for all prosecutor analyses"""
    
    def __init__(self, data_loader):
        self.loader = data_loader
        self.df_survey = data_loader.df_survey
        self.df_election = data_loader.df_election
        self.questions_df = data_loader.questions_df
        self.notable_cols = data_loader.notable_cols
        self.state_da_cols = data_loader.state_da_cols
        
        # Results storage
        self.results = {}
        
    def analyze_prosecutor_column(self, col, col_name=None):
        """Analyze a single prosecutor column"""
        if col_name is None:
            col_name = col
        
        responses = self.df_survey[col].dropna()
        total_responses = len(self.df_survey)
        
        # Convert to numeric
        numeric_ratings = responses.map(Config.RATING_MAP)
        substantive_ratings = numeric_ratings.dropna()
        
        if len(substantive_ratings) == 0:
            return None
        
        result = {
            'column': col,
            'name': col_name,
            'total_responses': total_responses,
            'substantive_ratings': len(substantive_ratings),
            'familiarity_rate': (len(substantive_ratings) / total_responses) * 100,
            'mean_score': substantive_ratings.mean(),
            'median_score': substantive_ratings.median(),
            'std_score': substantive_ratings.std(),
        }
        
        return result
    
    def build_prosecutor_dataset(self):
        """Build comprehensive prosecutor dataset"""
        print_section("BUILDING PROSECUTOR DATASET")
        
        all_prosecutors = []
        
        # Process notable prosecutors
        for col in self.notable_cols:
            if col in Config.DA_NAMES:
                name, location = Config.DA_NAMES[col]
                result = self.analyze_prosecutor_column(col, name)
                if result:
                    result['location'] = location
                    result['is_notable'] = True
                    result['name'] = name
                    all_prosecutors.append(result)
        
        # Process state prosecutors  
        for col in self.state_da_cols:
            if col in self.questions_df.columns:
                question_text = self.questions_df[col].iloc[0]
                if ' - ' in question_text:
                    name = question_text.split(' - ')[-1].strip()
                    result = self.analyze_prosecutor_column(col, name)
                    if result:
                        result['location'] = 'State'
                        result['is_notable'] = False
                        result['name'] = name
                        all_prosecutors.append(result)
        
        self.df_prosecutors = pd.DataFrame(all_prosecutors)
        print(f"✓ Created dataset with {len(self.df_prosecutors)} prosecutors")
        print(f"  - {len(self.df_prosecutors[self.df_prosecutors['is_notable']])} notable")
        print(f"  - {len(self.df_prosecutors[~self.df_prosecutors['is_notable']])} state")
        
        # Filter for meaningful IDEOLOGY ratings (≥10 for reliable mean scores)
        self.df_prosecutors_filtered_ideology = self.df_prosecutors[
            self.df_prosecutors['substantive_ratings'] >= Config.MIN_RATINGS_THRESHOLD
        ].copy()
        print(f"✓ Filtered to {len(self.df_prosecutors_filtered_ideology)} prosecutors with ≥{Config.MIN_RATINGS_THRESHOLD} ratings")
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
        
        # Add recalled indicator
        self.df_matched['recalled'] = self.df_matched['name'].isin(Config.RECALLED_PROSECUTORS)
        self.df_matched_all['recalled'] = self.df_matched_all['name'].isin(Config.RECALLED_PROSECUTORS)
        
        self.results['matched_all'] = self.df_matched_all
        self.results['matched_filtered'] = self.df_matched
        
        return self
    
    def analyze_familiarity_patterns(self):
        """Analyze familiarity patterns"""
        print_section("ANALYZING FAMILIARITY PATTERNS")
        
        # Notable vs state comparison
        notable = self.df_prosecutors[self.df_prosecutors['is_notable']]
        state = self.df_prosecutors[~self.df_prosecutors['is_notable']]
        
        stats = {
            'notable_mean_familiarity': notable['familiarity_rate'].mean(),
            'state_mean_familiarity': state['familiarity_rate'].mean(),
            'notable_unfamiliarity': 100 - notable['familiarity_rate'].mean(),
            'state_unfamiliarity': 100 - state['familiarity_rate'].mean(),
        }
        
        print(f"Notable prosecutors: {stats['notable_mean_familiarity']:.1f}% mean familiarity")
        print(f"                     {stats['notable_unfamiliarity']:.1f}% unfamiliarity")
        print(f"State prosecutors: {stats['state_mean_familiarity']:.1f}% mean familiarity")
        print(f"                   {stats['state_unfamiliarity']:.1f}% unfamiliarity")
        
        # Correlation with progressiveness
        filtered = self.df_prosecutors_filtered
        corr, p_val = pearsonr(filtered['mean_score'], filtered['familiarity_rate'])
        print(f"\nCorrelation (progressiveness × familiarity): r={corr:.3f}, p={p_val:.4f}")
        
        self.results['familiarity_stats'] = stats
        
        return self
    
    def analyze_incumbency(self):
        """Analyze incumbency advantage"""
        print_section("ANALYZING INCUMBENCY ADVANTAGE")
        
        if len(self.df_matched) < 10:
            print("Insufficient data for incumbency analysis")
            return self
        
        # All prosecutors
        all_incumb = self.df_matched[self.df_matched['ever_ran_as_incumbent']]
        all_chall = self.df_matched[self.df_matched['ever_ran_as_challenger']]
        
        print("ALL PROSECUTORS (Notable + State):")
        print(f"  Incumbents (n={len(all_incumb)}): {all_incumb['familiarity_rate'].mean():.2f}% mean familiarity")
        print(f"  Challengers (n={len(all_chall)}): {all_chall['familiarity_rate'].mean():.2f}% mean familiarity")
        
        if len(all_incumb) > 0 and len(all_chall) > 0:
            t, p = ttest_ind(all_incumb['familiarity_rate'], all_chall['familiarity_rate'])
            print(f"  T-test: t={t:.3f}, p={p:.4f}")
        
        # State-level only
        state_only = self.df_matched[~self.df_matched['is_notable']]
        if len(state_only) > 5:
            state_incumb = state_only[state_only['ever_ran_as_incumbent']]
            state_chall = state_only[state_only['ever_ran_as_challenger']]
            
            print("\nSTATE-LEVEL ONLY (Excluding Notable):")
            print(f"  Incumbents (n={len(state_incumb)}): {state_incumb['familiarity_rate'].mean():.2f}% mean familiarity")
            print(f"  Challengers (n={len(state_chall)}): {state_chall['familiarity_rate'].mean():.2f}% mean familiarity")
            
            if len(state_incumb) > 0 and len(state_chall) > 0:
                t, p = ttest_ind(state_incumb['familiarity_rate'], state_chall['familiarity_rate'])
                print(f"  T-test: t={t:.3f}, p={p:.4f}")
        
        return self
    
    def analyze_contestation(self):
        """Analyze electoral contestation patterns"""
        print_section("ANALYZING ELECTORAL CONTESTATION")
        
        progressive = self.df_matched[self.df_matched['ideology_category'] == 'Progressive']
        traditional = self.df_matched[self.df_matched['ideology_category'] == 'Traditional']
        
        if len(progressive) > 0 and len(traditional) > 0:
            print(f"Progressive prosecutors (n={len(progressive)}):")
            print(f"  Ever contested primary: {progressive['ever_contested_primary'].mean()*100:.1f}%")
            print(f"  Ever contested general: {progressive['ever_contested_general'].mean()*100:.1f}%")
            print()
            
            print(f"Traditional prosecutors (n={len(traditional)}):")
            print(f"  Ever contested primary: {traditional['ever_contested_primary'].mean()*100:.1f}%")
            print(f"  Ever contested general: {traditional['ever_contested_general'].mean()*100:.1f}%")
            print()
            
            # Chi-square test
            if len(progressive) > 5 and len(traditional) > 5:
                contingency = pd.crosstab(
                    self.df_matched[self.df_matched['ideology_category'].isin(['Progressive', 'Traditional'])]['ideology_category'],
                    self.df_matched[self.df_matched['ideology_category'].isin(['Progressive', 'Traditional'])]['ever_contested_general']
                )
                try:
                    chi2, p, _, _ = chi2_contingency(contingency)
                    print(f"Chi-square test (contested general): χ²={chi2:.3f}, p={p:.4f}")
                except:
                    print("Insufficient data for chi-square test")
        
        return self
    
    def analyze_recall_risk(self):
        """Analyze recall risk and close victory vulnerability"""
        print_section("ANALYZING RECALL RISK")
        
        recalled = self.df_matched[self.df_matched['recalled']]
        not_recalled = self.df_matched[~self.df_matched['recalled'] & self.df_matched['is_notable']]
        
        if len(recalled) > 0 and len(not_recalled) > 0:
            print("RECALLED PROSECUTORS:")
            for _, row in recalled.iterrows():
                print(f"\n{row['name']}:")
                if pd.notna(row['closest_general_margin']):
                    print(f"  Closest general margin: {row['closest_general_margin']:.1f}%")
                if pd.notna(row['closest_primary_margin']):
                    print(f"  Closest primary margin: {row['closest_primary_margin']:.1f}%")
            
            print("\n\nNON-RECALLED NOTABLE PROSECUTORS:")
            print(f"  Mean general margin: {not_recalled['closest_general_margin'].mean():.1f}%")
            print(f"  Median general margin: {not_recalled['closest_general_margin'].median():.1f}%")
            
            # Statistical test
            recalled_margins = recalled['closest_general_margin'].dropna()
            not_recalled_margins = not_recalled['closest_general_margin'].dropna()
            
            if len(recalled_margins) > 0 and len(not_recalled_margins) > 0:
                t, p = ttest_ind(recalled_margins, not_recalled_margins)
                print(f"\nT-test on general margins: t={t:.3f}, p={p:.4f}")
                if p < 0.05:
                    print("*** SIGNIFICANT DIFFERENCE ***")
        
        return self
    
    def run_multivariate_models(self):
        """Run comprehensive multivariate models"""
        print_section("MULTIVARIATE MODELING")
        
        if len(self.df_matched) < 20:
            print("Insufficient data for multivariate modeling")
            return self
        
        # Model 1: Familiarity model
        print("MODEL: Predictors of Familiarity")
        print("-" * 60)
        
        model_data = self.df_matched[[
            'familiarity_rate',
            'mean_score',
            'ever_contested_general',
            'ever_ran_as_challenger',
            'num_elections',
            'is_notable',
            'ever_ran_as_incumbent'
        ]].dropna()
        
        if len(model_data) > 20:
            X = model_data[[
                'mean_score',
                'ever_contested_general',
                'ever_ran_as_challenger',
                'num_elections',
                'is_notable',
                'ever_ran_as_incumbent'
            ]].astype(float)
            y = model_data['familiarity_rate']
            
            X_const = sm.add_constant(X)
            model = sm.OLS(y, X_const).fit()
            
            print(model.summary2().tables[1])
            print()
            
            print("KEY FINDINGS:")
            for var in X.columns:
                coef = model.params[var]
                p = model.pvalues[var]
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                print(f"  {var}: β={coef:.3f} {sig}")
            
            self.results['familiarity_model'] = model
        
        return self
    
    def export_results(self, output_dir):
        """Export all results to CSV"""
        print_section("EXPORTING RESULTS")
        
        ensure_dir(output_dir)
        
        # Export main datasets
        self.df_prosecutors.to_csv(f'{output_dir}/all_prosecutors.csv', index=False)
        print(f"✓ Saved all_prosecutors.csv")
        
        self.df_prosecutors_filtered.to_csv(f'{output_dir}/prosecutors_filtered.csv', index=False)
        print(f"✓ Saved prosecutors_filtered.csv")
        
        if hasattr(self, 'df_matched'):
            self.df_matched.to_csv(f'{output_dir}/matched_prosecutors.csv', index=False)
            print(f"✓ Saved matched_prosecutors.csv")
        
        return self


# ================================================================================
# VISUALIZATION CLASS
# ================================================================================

class Visualizer:
    """Create all visualizations"""
    
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.df_prosecutors = analyzer.df_prosecutors
        self.df_matched = analyzer.df_matched if hasattr(analyzer, 'df_matched') else None
        
    def create_all_visualizations(self, viz_dir):
        """Create all visualizations"""
        print_section("CREATING VISUALIZATIONS")
        
        ensure_dir(viz_dir)
        
        self.viz_top_prosecutors(viz_dir)
        self.viz_familiarity_comparison(viz_dir)
        self.viz_contestation(viz_dir)
        self.viz_recall_risk(viz_dir)
        self.viz_familiarity_model(viz_dir)
        
        print(f"\n✓ All visualizations saved to {viz_dir}/")
        
        return self
    
    def viz_top_prosecutors(self, viz_dir):
        """Visualize top progressive and traditional prosecutors"""
        filtered = self.df_prosecutors[
            self.df_prosecutors['substantive_ratings'] >= Config.MIN_RATINGS_THRESHOLD
        ].sort_values('mean_score')
        
        if len(filtered) < 10:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Most progressive
        top_prog = filtered.tail(10)
        ax1.barh(range(len(top_prog)), top_prog['mean_score'], color='#2E86AB')
        ax1.set_yticks(range(len(top_prog)))
        ax1.set_yticklabels([f"{row['name']}\n({row['familiarity_rate']:.1f}% familiar)" 
                              for _, row in top_prog.iterrows()], fontsize=9)
        ax1.set_xlabel('Mean Ideology Score', fontweight='bold')
        ax1.set_title('Top 10 Most Progressive Prosecutors', fontweight='bold', fontsize=14)
        ax1.set_xlim(1, 4)
        ax1.grid(axis='x', alpha=0.3)
        
        # Most traditional
        top_trad = filtered.head(10)
        ax2.barh(range(len(top_trad)), top_trad['mean_score'], color='#A23B72')
        ax2.set_yticks(range(len(top_trad)))
        ax2.set_yticklabels([f"{row['name']}\n({row['familiarity_rate']:.1f}% familiar)" 
                              for _, row in top_trad.iterrows()], fontsize=9)
        ax2.set_xlabel('Mean Ideology Score', fontweight='bold')
        ax2.set_title('Top 10 Most Traditional Prosecutors', fontweight='bold', fontsize=14)
        ax2.set_xlim(1, 4)
        ax2.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{viz_dir}/1_top_prosecutors.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Created top prosecutors visualization")
    
    def viz_familiarity_comparison(self, viz_dir):
        """Visualize familiarity patterns"""
        notable = self.df_prosecutors[self.df_prosecutors['is_notable']]
        state = self.df_prosecutors[~self.df_prosecutors['is_notable']]
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        data = [notable['familiarity_rate'], state['familiarity_rate']]
        bp = ax.boxplot(data, labels=['Nationally Notable', 'State-Level'], patch_artist=True)
        
        for patch, color in zip(bp['boxes'], ['lightcoral', 'lightblue']):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel('Familiarity Rate (%)', fontweight='bold', fontsize=12)
        ax.set_title('Prosecutor Familiarity: Notable vs. State-Level', fontweight='bold', fontsize=14)
        ax.grid(axis='y', alpha=0.3)
        
        # Add means
        means = [d.mean() for d in data]
        ax.plot([1, 2], means, 'ro-', linewidth=2, markersize=8, label='Mean')
        
        for i, mean in enumerate(means, 1):
            ax.text(i, mean + 2, f'{mean:.1f}%', ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{viz_dir}/2_familiarity_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Created familiarity comparison visualization")
    
    def viz_contestation(self, viz_dir):
        """Visualize contestation patterns"""
        if self.df_matched is None or len(self.df_matched) < 10:
            return
        
        prog = self.df_matched[self.df_matched['ideology_category'] == 'Progressive']
        trad = self.df_matched[self.df_matched['ideology_category'] == 'Traditional']
        
        if len(prog) == 0 or len(trad) == 0:
            return
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        categories = ['Contested\nPrimary', 'Contested\nGeneral']
        progressive_rates = [
            prog['ever_contested_primary'].mean() * 100,
            prog['ever_contested_general'].mean() * 100
        ]
        traditional_rates = [
            trad['ever_contested_primary'].mean() * 100,
            trad['ever_contested_general'].mean() * 100
        ]
        
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, progressive_rates, width, label='Progressive', color='#2E86AB')
        bars2 = ax.bar(x + width/2, traditional_rates, width, label='Traditional', color='#A23B72')
        
        ax.set_ylabel('% Ever Contested', fontweight='bold', fontsize=12)
        ax.set_title('Contestation Rates by Ideology', fontweight='bold', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend(fontsize=11)
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                        f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f'{viz_dir}/3_contestation_patterns.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Created contestation visualization")
    
    def viz_recall_risk(self, viz_dir):
        """Visualize recall risk patterns"""
        if self.df_matched is None:
            return
        
        recalled = self.df_matched[self.df_matched['recalled']]
        not_recalled = self.df_matched[~self.df_matched['recalled'] & self.df_matched['is_notable']]
        
        if len(recalled) == 0 or len(not_recalled) == 0:
            return
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        recalled_margins = recalled['closest_general_margin'].dropna().tolist()
        recalled_names = recalled[recalled['closest_general_margin'].notna()]['name'].tolist()
        not_recalled_margins = not_recalled['closest_general_margin'].dropna()
        
        # Scatter plot
        ax.scatter([1]*len(recalled_margins), recalled_margins, s=200, c='red', 
                   alpha=0.7, edgecolors='darkred', linewidth=2, label='Recalled', zorder=3)
        ax.scatter([2]*len(not_recalled_margins), not_recalled_margins, s=100, c='blue', 
                   alpha=0.5, edgecolors='darkblue', linewidth=1, label='Not Recalled', zorder=2)
        
        # Add means
        mean_recalled = np.mean(recalled_margins)
        mean_not_recalled = not_recalled_margins.mean()
        
        ax.plot([0.8, 1.2], [mean_recalled, mean_recalled], 'r-', linewidth=3, 
                label=f'Recalled Mean: {mean_recalled:.1f}%')
        ax.plot([1.8, 2.2], [mean_not_recalled, mean_not_recalled], 'b-', linewidth=3, 
                label=f'Not Recalled Mean: {mean_not_recalled:.1f}%')
        
        # Add names
        for margin, name in zip(recalled_margins, recalled_names):
            ax.text(1.05, margin, name, fontsize=10, va='center')
        
        ax.set_xlim(0.5, 2.5)
        ax.set_ylim(20, 100)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Recalled\nProsecutors', 'Not Recalled\nNotable Prosecutors'], fontsize=12)
        ax.set_ylabel('Closest General Election Margin (%)', fontweight='bold', fontsize=12)
        ax.set_title('Initial Election Margins: Recalled vs. Not Recalled', fontweight='bold', fontsize=14)
        ax.axhline(y=55, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='55% threshold')
        ax.legend(fontsize=10, loc='lower right')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{viz_dir}/4_recall_risk.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Created recall risk visualization")
    
    def viz_familiarity_model(self, viz_dir):
        """Visualize familiarity model results"""
        if 'familiarity_model' not in self.analyzer.results:
            return
        
        model = self.analyzer.results['familiarity_model']
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        
        predictors = ['Progressiveness', 'Contested\nGeneral', 'Ran as\nChallenger', 
                      'Number of\nElections', 'Being\nNotable', 'Incumbency']
        
        # Get coefficients (excluding constant)
        coefficients = []
        p_values = []
        param_names = ['mean_score', 'ever_contested_general', 'ever_ran_as_challenger',
                       'num_elections', 'is_notable', 'ever_ran_as_incumbent']
        
        for param in param_names:
            if param in model.params:
                coefficients.append(model.params[param])
                p_values.append(model.pvalues[param])
        
        colors = ['green' if p < 0.05 else 'gray' for p in p_values]
        
        y_pos = np.arange(len(predictors))
        bars = ax.barh(y_pos, coefficients, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        
        ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(predictors, fontsize=12)
        ax.set_xlabel('Coefficient (Percentage Points)', fontweight='bold', fontsize=12)
        ax.set_title('Predictors of Prosecutor Familiarity\n(Multivariate OLS Regression)', 
                     fontweight='bold', fontsize=14)
        
        # Add value labels
        for i, (bar, coef, p) in enumerate(zip(bars, coefficients, p_values)):
            stars = '**' if p < 0.01 else '*' if p < 0.05 else ''
            x_pos = coef + (0.5 if coef > 0 else -0.5)
            ax.text(x_pos, i, f'{coef:.2f}{stars}', 
                    va='center', ha='left' if coef > 0 else 'right', fontsize=11, fontweight='bold')
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', alpha=0.7, edgecolor='black', label='Significant (p < 0.05)'),
            Patch(facecolor='gray', alpha=0.7, edgecolor='black', label='Not Significant')
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
        
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{viz_dir}/5_familiarity_model.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Created familiarity model visualization")


# ================================================================================
# MAIN EXECUTION
# ================================================================================

def main():
    """Main execution function"""
    
    print("\n" + "="*80)
    print("PROSECUTOR IDEOLOGY SURVEY - COMPREHENSIVE ANALYSIS")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Survey file: {Config.SURVEY_FILE}")
    print(f"  Election file: {Config.ELECTION_FILE}")
    print(f"  Output directory: {Config.OUTPUT_DIR}")
    print(f"  Min ratings threshold: {Config.MIN_RATINGS_THRESHOLD}")
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
    
    # Create visualizations
    visualizer = Visualizer(analyzer)
    visualizer.create_all_visualizations(Config.VIZ_DIR)
    
    print_section("✅ ANALYSIS COMPLETE!", "=")
    print(f"All outputs saved to: {Config.OUTPUT_DIR}/")
    print(f"All visualizations saved to: {Config.VIZ_DIR}/")
    print("\nGenerated files:")
    print("  📊 Data Files:")
    print("     - all_prosecutors.csv")
    print("     - prosecutors_filtered.csv")
    print("     - matched_prosecutors.csv")
    print("\n  📈 Visualizations:")
    print("     - 1_top_prosecutors.png")
    print("     - 2_familiarity_comparison.png")
    print("     - 3_contestation_patterns.png")
    print("     - 4_recall_risk.png")
    print("     - 5_familiarity_model.png")
    print("\n" + "="*80)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
COMPREHENSIVE District Attorney Progressiveness Survey Analysis
================================================================
This script combines the best features from multiple analysis approaches:
- Complete visualization suite from the original analysis
- Detailed familiarity tracking (explicit vs blank responses)
- Open-text theme analysis
- Object-oriented design for better organization
- State-level and national prosecutor comparisons
- Position-based effects analysis
- Electoral transition tracking

Author: Dvir Yogev, BERQ-J
Date: October 2025
Scale: 1-4 (Very Traditional=1, Traditional=2, Progressive=3, Very Progressive=4)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr
import statsmodels.api as sm
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set up plotting parameters
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = True
sns.set_style("whitegrid")


# =============================================================================
# COMPREHENSIVE ANALYZER CLASS
# =============================================================================

class ComprehensiveProsecutorAnalyzer:
    """
    Comprehensive analyzer combining all analysis features:
    - Detailed familiarity tracking
    - Position-based analysis
    - Geographic patterns
    - Electoral transitions
    - Open-text analysis
    - Complete visualization suite
    """
    
    def __init__(self, survey_file, min_ratings_threshold=10):
        """
        Initialize the comprehensive analyzer
        
        Parameters:
        -----------
        survey_file : str
            Path to the Qualtrics survey CSV file
        min_ratings_threshold : int
            Minimum number of ratings needed to include a prosecutor
        """
        self.survey_file = survey_file
        self.min_ratings = min_ratings_threshold
        
        # CRITICAL: Rating scale is 1-4 (NOT 1-5)
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
        """Load and prepare the survey data"""
        print("=" * 80)
        print("LOADING DATA")
        print("=" * 80)
        
        # Try loading with skiprows first (Qualtrics format)
        try:
            self.df = pd.read_csv(self.survey_file, skiprows=[1])
            self.questions_df = pd.read_csv(self.survey_file, nrows=1)
            print(f"✓ Loaded {len(self.df)} survey responses (Qualtrics format)")
        except:
            # Fall back to standard loading
            self.df = pd.read_csv(self.survey_file)
            self.questions_df = None
            print(f"✓ Loaded {len(self.df)} survey responses (standard format)")
        
        # Filter for consent
        if 'con' in self.df.columns:
            self.df_consented = self.df[self.df['con'] == 'Agree'].copy()
            print(f"✓ After consent filter: {len(self.df_consented)} respondents")
        else:
            self.df_consented = self.df.copy()
            print(f"✓ No consent column found, using all {len(self.df_consented)} respondents")
        
        # Identify columns
        self.notable_cols = [col for col in self.df_consented.columns if col.startswith('notable_')]
        print(f"✓ Identified {len(self.notable_cols)} notable prosecutors")
        
        # Identify state DA columns
        self.state_da_cols = []
        for col in self.df_consented.columns:
            if '_' in col and not col.startswith('Q') and not col.startswith('notable'):
                if col not in ['gender_4_TEXT', 'race_1', 'race_2', 'race_3', 'race_4', 
                              'race_5', 'race_6', 'race_7', 'Q55_5_TEXT']:
                    unique_vals = self.df_consented[col].dropna().unique()
                    if len(unique_vals) > 0 and any('Progressive' in str(v) or 'Traditional' in str(v) 
                                                     for v in unique_vals):
                        self.state_da_cols.append(col)
        
        print(f"✓ Identified {len(self.state_da_cols)} state prosecutor columns")
        print()
    
    def get_prosecutor_name(self, col):
        """Extract prosecutor name from question text or mapping"""
        if col in self.da_names:
            return self.da_names[col]
        elif self.questions_df is not None and col in self.questions_df.columns:
            question_text = self.questions_df[col].iloc[0]
            if ' - ' in question_text:
                return question_text.split(' - ')[-1].strip()
        return 'Unknown'
    
    # =========================================================================
    # PART 1: FAMILIARITY ANALYSIS (Enhanced from executive summary script)
    # =========================================================================
    
    def analyze_familiarity(self):
        """
        Comprehensive familiarity analysis distinguishing between:
        - Explicit "Not Familiar" responses
        - Left blank (implicit unfamiliarity)
        - Substantive ratings
        """
        print("=" * 80)
        print("PART 1: COMPREHENSIVE FAMILIARITY ANALYSIS")
        print("=" * 80)
        
        # Notable prosecutors familiarity
        rated_at_least_one = self.df_consented[self.notable_cols].notna().any(axis=1)
        respondents_who_rated = rated_at_least_one.sum()
        
        total_opportunities = respondents_who_rated * len(self.notable_cols)
        explicit_not_familiar = 0
        left_blank = 0
        substantive = 0
        
        for idx, row in self.df_consented[rated_at_least_one].iterrows():
            for col in self.notable_cols:
                val = row[col]
                if pd.isna(val):
                    left_blank += 1
                elif val == 'Not Familiar':
                    explicit_not_familiar += 1
                else:
                    substantive += 1
        
        total_unfamiliar = explicit_not_familiar + left_blank
        
        print(f"\n📊 NOTABLE PROSECUTORS (National Sample):")
        print(f"   Respondents who rated: {respondents_who_rated}")
        print(f"   Total opportunities: {total_opportunities:,}")
        print(f"   Explicit 'Not Familiar': {explicit_not_familiar:,} ({explicit_not_familiar/total_opportunities*100:.1f}%)")
        print(f"   Left blank: {left_blank:,} ({left_blank/total_opportunities*100:.1f}%)")
        print(f"   Total unfamiliar: {total_unfamiliar:,} ({total_unfamiliar/total_opportunities*100:.1f}%)")
        print(f"   Substantive ratings: {substantive:,} ({substantive/total_opportunities*100:.1f}%)")
        
        # Completion pattern
        ratings_per_respondent = []
        for idx, row in self.df_consented[rated_at_least_one].iterrows():
            count = sum(1 for col in self.notable_cols if pd.notna(row[col]))
            ratings_per_respondent.append(count)
        
        rated_all_50 = sum(1 for x in ratings_per_respondent if x == 50)
        print(f"   Completed all 50 prosecutors: {rated_all_50} ({rated_all_50/respondents_who_rated*100:.1f}%)")
        
        # State prosecutors familiarity (if applicable)
        if 'more' in self.df_consented.columns:
            agreed_to_rate_state = self.df_consented['more'] == 'Show me the prosecutors from my state'
            
            state_to_cols = {}
            for col in self.state_da_cols:
                state_name = col.rsplit('_', 1)[0]
                if state_name not in state_to_cols:
                    state_to_cols[state_name] = []
                state_to_cols[state_name].append(col)
            
            state_total_opp = 0
            state_explicit_nf = 0
            state_blank = 0
            state_substantive = 0
            
            for idx, row in self.df_consented[agreed_to_rate_state].iterrows():
                respondent_state = row.get('Q1', None)
                if pd.isna(respondent_state):
                    continue
                state_cols = state_to_cols.get(respondent_state, [])
                for col in state_cols:
                    state_total_opp += 1
                    val = row[col]
                    if pd.isna(val):
                        state_blank += 1
                    elif val == 'Not Familiar':
                        state_explicit_nf += 1
                    else:
                        state_substantive += 1
            
            if state_total_opp > 0:
                state_total_unfam = state_explicit_nf + state_blank
                
                print(f"\n📊 STATE PROSECUTORS (Within-State Sample):")
                print(f"   Agreed to rate state DAs: {agreed_to_rate_state.sum()}")
                print(f"   Total opportunities: {state_total_opp:,}")
                print(f"   Explicit 'Not Familiar': {state_explicit_nf:,} ({state_explicit_nf/state_total_opp*100:.1f}%)")
                print(f"   Left blank: {state_blank:,} ({state_blank/state_total_opp*100:.1f}%)")
                print(f"   Total unfamiliar: {state_total_unfam:,} ({state_total_unfam/state_total_opp*100:.1f}%)")
                print(f"   Substantive ratings: {state_substantive:,} ({state_substantive/state_total_opp*100:.1f}%)")
        
        # Store familiarity metrics for later use
        self.familiarity_metrics = {
            'total_opportunities': total_opportunities,
            'explicit_not_familiar': explicit_not_familiar,
            'left_blank': left_blank,
            'substantive': substantive
        }
        
        print()
    
    # =========================================================================
    # PART 2: NOTABLE PROSECUTOR ANALYSIS (50 National DAs)
    # =========================================================================
    
    def analyze_notable_prosecutors(self):
        """Analyze all 50 notable prosecutors with comprehensive metrics"""
        print("=" * 80)
        print("PART 2: NOTABLE PROSECUTOR RANKINGS (50 National DAs)")
        print("=" * 80)
        
        # Positions for analysis
        positions = [
            'Academic (e.g., law professor, criminal justice researcher)',
            'Defense Attorney',
            'Prosecutor'
        ]
        
        # Build comprehensive dataset for all DAs
        da_data = []
        for col, full_name in self.da_names.items():
            name, location = full_name.split('|')
            
            if col in self.df_consented.columns:
                ratings = self.df_consented[col].dropna()
                numeric = ratings.map(self.rating_map).dropna()
                
                # Calculate position-specific means
                position_means = {}
                for position in positions:
                    pos_df = self.df_consented[self.df_consented['position'] == position]
                    pos_ratings = pos_df[col].map(self.rating_map).dropna()
                    if len(pos_ratings) >= 5:
                        position_means[position.split('(')[0].strip()] = pos_ratings.mean()
                    else:
                        position_means[position.split('(')[0].strip()] = np.nan
                
                # Count responses by category
                response_counts = ratings.value_counts()
                
                da_data.append({
                    'Name': name,
                    'Location': location,
                    'Column': col,
                    'Total_Ratings': len(ratings),
                    'Substantive_Ratings': len(numeric),
                    'Familiarity_Rate': len(numeric)/len(ratings)*100 if len(ratings) > 0 else 0,
                    'Mean_Score': numeric.mean() if len(numeric) > 0 else np.nan,
                    'Median_Score': numeric.median() if len(numeric) > 0 else np.nan,
                    'Std_Dev': numeric.std() if len(numeric) > 0 else np.nan,
                    'Very_Traditional': response_counts.get('Very Traditional', 0),
                    'Traditional': response_counts.get('Traditional', 0),
                    'Progressive': response_counts.get('Progressive', 0),
                    'Very_Progressive': response_counts.get('Very Progressive', 0),
                    'Not_Familiar': response_counts.get('Not Familiar', 0),
                    'Very_Progressive_Pct': (numeric == 4).sum() / len(numeric) * 100 if len(numeric) > 0 else 0,
                    'Progressive_Pct': (numeric == 3).sum() / len(numeric) * 100 if len(numeric) > 0 else 0,
                    'Traditional_Pct': (numeric == 2).sum() / len(numeric) * 100 if len(numeric) > 0 else 0,
                    'Very_Traditional_Pct': (numeric == 1).sum() / len(numeric) * 100 if len(numeric) > 0 else 0,
                    'Academic_Mean': position_means.get('Academic', np.nan),
                    'Defense_Mean': position_means.get('Defense Attorney', np.nan),
                    'Prosecutor_Mean': position_means.get('Prosecutor', np.nan)
                })
        
        self.da_df = pd.DataFrame(da_data)
        self.da_df = self.da_df.sort_values('Mean_Score', ascending=False)
        
        # Filter for reliable ratings
        reliable = self.da_df[self.da_df['Substantive_Ratings'] >= self.min_ratings].copy()
        
        print(f"\n✓ Analyzed {len(self.da_df)} prosecutors")
        print(f"✓ {len(reliable)} with ≥{self.min_ratings} substantive ratings")
        
        # Print top progressive
        print(f"\n🏆 MOST PROGRESSIVE:")
        for idx, (i, row) in enumerate(reliable.head(5).iterrows(), 1):
            print(f"   {idx}. {row['Name']} ({row['Location']})")
            print(f"      Mean: {row['Mean_Score']:.2f} | N: {row['Substantive_Ratings']} | "
                  f"Very Prog: {row['Very_Progressive_Pct']:.0f}%")
        
        # Print most traditional
        print(f"\n🏛️  MOST TRADITIONAL:")
        for idx, (i, row) in enumerate(reliable.tail(5).sort_values('Mean_Score').iterrows(), 1):
            print(f"   {idx}. {row['Name']} ({row['Location']})")
            print(f"      Mean: {row['Mean_Score']:.2f} | N: {row['Substantive_Ratings']} | "
                  f"Very Trad: {row['Very_Traditional_Pct']:.0f}%")
        
        # Ideological distribution
        prog_count = len(reliable[reliable['Mean_Score'] >= 3])
        trad_count = len(reliable[reliable['Mean_Score'] <= 2])
        mod_count = len(reliable[(reliable['Mean_Score'] > 2) & (reliable['Mean_Score'] < 3)])
        
        print(f"\n📊 IDEOLOGICAL DISTRIBUTION:")
        print(f"   Progressive (≥3): {prog_count} ({prog_count/len(reliable)*100:.0f}%)")
        print(f"   Moderate (>2 and <3): {mod_count} ({mod_count/len(reliable)*100:.0f}%)")
        print(f"   Traditional (≤2): {trad_count} ({trad_count/len(reliable)*100:.0f}%)")
        print(f"   Overall Mean: {reliable['Mean_Score'].mean():.2f} [Midpoint: 2.5]")
        
        print()
    
    # =========================================================================
    # PART 3: JURISDICTION TRANSITION ANALYSIS
    # =========================================================================
    
    def analyze_jurisdiction_transitions(self):
        """Analyze transitions between prosecutors in same jurisdiction"""
        print("=" * 80)
        print("PART 3: JURISDICTION TRANSITION ANALYSIS")
        print("=" * 80)
        
        transitions = [
            ('notable_5', 'notable_4', 'San Francisco, CA'),    # Boudin → Jenkins
            ('notable_7', 'notable_6', 'Alameda County, CA'),   # O'Malley → Price
            ('notable_8', 'notable_9', 'Los Angeles, CA'),      # Gascon → Hochman
            ('notable_10', 'notable_11', 'Cook County, IL'),    # Foxx → Burke
            ('notable_32', 'notable_33', 'King County, WA'),    # Satterberg → Manion
            ('notable_29', 'notable_30', 'Suffolk County, MA'), # Rollins → Hayden
            ('notable_37', 'notable_38', 'Multnomah County, OR'), # Schmidt → Vasquez
            ('notable_39', 'notable_40', 'Shelby County, TN'),  # Weirich → Mulroy
            ('notable_26', 'notable_27', 'Harris County, TX'),  # Ogg → Teare
            ('notable_47', 'notable_48', 'Washington, DC'),     # Racine → Schwalb
            ('notable_49', 'notable_50', 'Baltimore City, MD')  # Mosby → Bates
        ]
        
        transition_results = []
        for predecessor_col, successor_col, jurisdiction in transitions:
            pred_name = self.da_names[predecessor_col].split('|')[0]
            succ_name = self.da_names[successor_col].split('|')[0]
            
            pred_row = self.da_df[self.da_df['Column'] == predecessor_col]
            succ_row = self.da_df[self.da_df['Column'] == successor_col]
            
            if len(pred_row) > 0 and len(succ_row) > 0:
                pred_score = pred_row['Mean_Score'].values[0]
                succ_score = succ_row['Mean_Score'].values[0]
                
                if not np.isnan(pred_score) and not np.isnan(succ_score):
                    change = succ_score - pred_score
                    direction = "→ Traditional" if change < 0 else "→ Progressive"
                    
                    transition_results.append({
                        'Jurisdiction': jurisdiction,
                        'Predecessor': pred_name,
                        'Successor': succ_name,
                        'Pred_Score': pred_score,
                        'Succ_Score': succ_score,
                        'Change': change,
                        'Direction': direction
                    })
        
        self.transitions_df = pd.DataFrame(transition_results).sort_values('Change')
        
        print(f"\n✓ Analyzed {len(self.transitions_df)} jurisdiction transitions\n")
        
        # Summary statistics
        shifts_traditional = len(self.transitions_df[self.transitions_df['Change'] < 0])
        shifts_progressive = len(self.transitions_df[self.transitions_df['Change'] > 0])
        avg_change = self.transitions_df['Change'].mean()
        
        print(f"📊 TRANSITION SUMMARY:")
        print(f"   Shifts toward traditional: {shifts_traditional}")
        print(f"   Shifts toward progressive: {shifts_progressive}")
        print(f"   Average change: {avg_change:+.2f}")
        
        print(f"\n🔄 LARGEST SHIFTS:")
        for idx, row in self.transitions_df.head(3).iterrows():
            print(f"   {row['Jurisdiction']}")
            print(f"      {row['Predecessor']} ({row['Pred_Score']:.2f}) → {row['Successor']} ({row['Succ_Score']:.2f})")
            print(f"      Change: {row['Change']:+.2f} {row['Direction']}")
        
        print()
    
    # =========================================================================
    # PART 4: STATE-LEVEL PATTERN ANALYSIS
    # =========================================================================
    
    def analyze_state_patterns(self):
        """Analyze state-level prosecutor patterns"""
        print("=" * 80)
        print("PART 4: STATE-LEVEL PATTERN ANALYSIS")
        print("=" * 80)
        
        if len(self.state_da_cols) == 0:
            print("\n⚠️  No state-level data available")
            print()
            return
        
        state_results = {}
        for col in self.state_da_cols:
            state = col.rsplit('_', 1)[0]
            ratings = self.df_consented[col].map(self.rating_map).dropna()
            
            if state not in state_results:
                state_results[state] = {
                    'ratings': [],
                    'prosecutors': 0
                }
            
            state_results[state]['prosecutors'] += 1
            state_results[state]['ratings'].extend(ratings.tolist())
        
        state_summary = []
        for state, data in state_results.items():
            if len(data['ratings']) >= 10:
                ratings_array = np.array(data['ratings'])
                state_summary.append({
                    'state': state,
                    'n_prosecutors': data['prosecutors'],
                    'total_ratings': len(ratings_array),
                    'mean_score': ratings_array.mean(),
                    'median_score': np.median(ratings_array),
                    'std_score': ratings_array.std(),
                    'progressive_pct': ((ratings_array >= 3).sum() / len(ratings_array) * 100)
                })
        
        self.state_df = pd.DataFrame(state_summary).sort_values('mean_score', ascending=False)
        
        print(f"\n✓ States with sufficient data (≥10 ratings): {len(self.state_df)}\n")
        
        if len(self.state_df) > 0:
            print("🏆 TOP 5 MOST PROGRESSIVE STATES:")
            for idx, (i, row) in enumerate(self.state_df.head(5).iterrows(), 1):
                print(f"   {idx}. {row['state']}: Mean={row['mean_score']:.2f}, "
                      f"N={row['total_ratings']}, Progressive={row['progressive_pct']:.0f}%")
            
            print("\n🏛️  TOP 5 MOST TRADITIONAL STATES:")
            for idx, (i, row) in enumerate(self.state_df.tail(5).sort_values('mean_score').iterrows(), 1):
                print(f"   {idx}. {row['state']}: Mean={row['mean_score']:.2f}, "
                      f"N={row['total_ratings']}, Progressive={row['progressive_pct']:.0f}%")
            
            # Calculate national vs state gap
            national_mean = self.da_df['Mean_Score'].mean()
            state_mean = self.state_df['mean_score'].mean()
            
            print(f"\n📊 NATIONAL VS STATE COMPARISON:")
            print(f"   National DAs mean: {national_mean:.2f}")
            print(f"   State DAs mean: {state_mean:.2f}")
            print(f"   Gap: {national_mean - state_mean:.2f} points")
        
        print()
    
    # =========================================================================
    # PART 5: POSITION-BASED ANALYSIS
    # =========================================================================
    
    def analyze_by_position(self):
        """Analyze rating patterns by respondent position"""
        print("=" * 80)
        print("PART 5: ANALYSIS BY PROFESSIONAL POSITION")
        print("=" * 80)
        
        position_counts = self.df_consented['position'].value_counts()
        print("\n📊 SAMPLE COMPOSITION:")
        for pos, count in position_counts.head(7).items():
            print(f"   {pos}: {count} ({count/len(self.df_consented)*100:.1f}%)")
        
        main_positions = [
            'Prosecutor',
            'Defense Attorney',
            'Academic (e.g., law professor, criminal justice researcher)'
        ]
        
        print("\n📈 RATING PATTERNS BY POSITION:")
        position_stats = []
        
        for position in main_positions:
            subset = self.df_consented[self.df_consented['position'] == position]
            
            all_ratings = []
            for col in self.notable_cols:
                ratings = subset[col].map(self.rating_map).dropna()
                all_ratings.extend(ratings.tolist())
            
            if len(all_ratings) > 0:
                ratings_array = np.array(all_ratings)
                
                stats_dict = {
                    'position': position,
                    'n_respondents': len(subset),
                    'total_ratings': len(all_ratings),
                    'mean_score': ratings_array.mean(),
                    'progressive_pct': (ratings_array >= 3).sum()/len(ratings_array)*100,
                    'traditional_pct': (ratings_array <= 2).sum()/len(ratings_array)*100
                }
                position_stats.append(stats_dict)
                
                print(f"\n   {position}:")
                print(f"      N respondents: {len(subset)}")
                print(f"      Total ratings: {len(all_ratings)}")
                print(f"      Mean score: {ratings_array.mean():.2f}")
                print(f"      % Progressive/Very Progressive: "
                      f"{(ratings_array >= 3).sum()/len(ratings_array)*100:.1f}%")
                print(f"      % Traditional/Very Traditional: "
                      f"{(ratings_array <= 2).sum()/len(ratings_array)*100:.1f}%")
        
        self.position_stats = pd.DataFrame(position_stats)
        print()
    
    # =========================================================================
    # PART 6: OPEN-TEXT ANALYSIS
    # =========================================================================
    
    def analyze_open_text(self):
        """Analyze open-text responses for themes"""
        print("=" * 80)
        print("PART 6: OPEN-TEXT RESPONSE ANALYSIS")
        print("=" * 80)
        
        # Q54: Progressive characteristics
        if 'Q54' in self.df_consented.columns:
            q54_responses = self.df_consented['Q54'].dropna()
            print(f"\n📝 Q54 (Progressive characteristics): {len(q54_responses)} responses")
            
            # Theme analysis
            themes = {
                'Rehabilitation/Treatment': ['rehab', 'treatment', 'diversion'],
                'Reduce Incarceration': ['reduce', 'decreas', 'incarceration', 'mass incarceration'],
                'Bail Reform': ['bail', 'pretrial'],
                'Racial Justice': ['racial', 'race', 'equity', 'disparit'],
                'Police Accountability': ['police', 'misconduct', 'officer'],
                'Transparency': ['transparent', 'accountab', 'data'],
                'Restorative Justice': ['restorative', 'victim', 'community'],
                'Sentencing Reform': ['sentenc', 'mandatory minimum']
            }
            
            print(f"\n   THEME FREQUENCY:")
            for theme, keywords in themes.items():
                count = sum(1 for r in q54_responses if any(kw in str(r).lower() for kw in keywords))
                pct = count / len(q54_responses) * 100
                print(f"      {theme}: {count} ({pct:.0f}%)")
        
        # Q55: Impact assessment
        if 'Q55' in self.df_consented.columns:
            q55_responses = self.df_consented['Q55'].value_counts()
            print(f"\n📊 Q55 (Impact on reform movement):")
            for val, count in q55_responses.items():
                pct = count / q55_responses.sum() * 100
                print(f"      {val}: {count} ({pct:.1f}%)")
        
        print()
    
    # =========================================================================
    # PART 7: GEOGRAPHIC DISTRIBUTION
    # =========================================================================
    
    def analyze_geographic_distribution(self):
        """Analyze geographic distribution of respondents"""
        print("=" * 80)
        print("PART 7: GEOGRAPHIC DISTRIBUTION")
        print("=" * 80)
        
        if 'Q1' in self.df_consented.columns:
            state_counts = self.df_consented['Q1'].value_counts().head(10)
            print("\n🗺️  TOP 10 STATES BY RESPONDENT LOCATION:")
            for state, count in state_counts.items():
                pct = count / len(self.df_consented) * 100
                print(f"   {state}: {count} ({pct:.1f}%)")
        elif 'RespondentState' in self.df_consented.columns:
            state_counts = self.df_consented['RespondentState'].value_counts().head(10)
            print("\n🗺️  TOP 10 STATES BY RESPONDENT LOCATION:")
            for state, count in state_counts.items():
                pct = count / len(self.df_consented) * 100
                print(f"   {state}: {count} ({pct:.1f}%)")
        else:
            print("\n⚠️  No geographic data available")
        
        print()
    
    # =========================================================================
    # PART 8: ADDITIONAL ANALYSES
    # =========================================================================
    
    def analyze_additional_metrics(self):
        """Additional specialized analyses"""
        print("=" * 80)
        print("PART 8: ADDITIONAL ANALYSES")
        print("=" * 80)
        
        # Weighted analysis by familiarity
        print("\n📊 WEIGHTED ANALYSIS BY FAMILIARITY:")
        self.da_df['Familiarity_Group'] = pd.cut(
            self.da_df['Familiarity_Rate'],
            bins=[0, 10, 20, 30, 100],
            labels=['Low (<10%)', 'Medium (10-20%)', 'High (20-30%)', 'Very High (>30%)']
        )
        
        for group in ['Very High (>30%)', 'High (20-30%)', 'Medium (10-20%)', 'Low (<10%)']:
            group_das = self.da_df[self.da_df['Familiarity_Group'] == group]
            if len(group_das) > 0:
                mean_score = group_das['Mean_Score'].mean()
                print(f"   {group}: {mean_score:.2f} (n={len(group_das)} DAs)")
        
        # Most controversial DAs (highest standard deviation)
        print("\n🔥 MOST CONTROVERSIAL DAs (Highest Std Dev):")
        controversial = self.da_df[self.da_df['Substantive_Ratings'] >= 30].copy()
        controversial = controversial.sort_values('Std_Dev', ascending=False)
        
        for idx, row in controversial.head(5).iterrows():
            print(f"   {row['Name']}: StdDev={row['Std_Dev']:.2f}, Mean={row['Mean_Score']:.2f}")
        
        # Position consensus vs disagreement
        print("\n🤝 POSITION-BASED CONSENSUS AND DISAGREEMENT:")
        self.da_df['Position_Variance'] = self.da_df[['Academic_Mean', 'Defense_Mean', 'Prosecutor_Mean']].var(axis=1)
        
        consensus = self.da_df[self.da_df['Substantive_Ratings'] >= 30].copy()
        
        print("\n   DAs with most position consensus (lowest variance):")
        consensus_sorted = consensus.sort_values('Position_Variance').head(3)
        for _, row in consensus_sorted.iterrows():
            print(f"      {row['Name']}: Variance={row['Position_Variance']:.3f}")
        
        print("\n   DAs with most position disagreement (highest variance):")
        disagreement_sorted = consensus.sort_values('Position_Variance', ascending=False).head(3)
        for _, row in disagreement_sorted.iterrows():
            print(f"      {row['Name']}: Variance={row['Position_Variance']:.3f}")
        
        print()
    
    # =========================================================================
    # PART 9: COMPREHENSIVE VISUALIZATION SUITE
    # =========================================================================
    
    def create_all_visualizations(self, output_dir='output'):
        """Create all visualizations from both scripts"""
        print("=" * 80)
        print("PART 9: CREATING COMPREHENSIVE VISUALIZATION SUITE")
        print("=" * 80)
        print()
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # VIZ 1: Top 20 Rankings
        self.create_viz_rankings(output_path)
        
        # VIZ 2: All 50 DAs overview
        self.create_viz_all_50(output_path)
        
        # VIZ 3: Position effects
        self.create_viz_position_effects(output_path)
        
        # VIZ 4: Familiarity crisis
        self.create_viz_familiarity(output_path)
        
        # VIZ 5: Electoral transitions
        if hasattr(self, 'transitions_df'):
            self.create_viz_transitions(output_path)
        
        # VIZ 6: Controversy analysis
        self.create_viz_controversy(output_path)
        
        # VIZ 7: Geographic distribution
        self.create_viz_geographic(output_path)
        
        # VIZ 8: National vs State comparison
        if hasattr(self, 'state_df') and len(self.state_df) > 0:
            self.create_viz_national_vs_state(output_path)
        
        print("\n✅ All visualizations created successfully!")
        print()
    
    def create_viz_rankings(self, output_path):
        """Create top 20 rankings visualization"""
        print("   Creating Viz 1: Top 20 Rankings...")
        
        reliable = self.da_df[self.da_df['Substantive_Ratings'] >= 30].copy()
        top_20 = reliable.head(20)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        y_pos = np.arange(len(top_20))
        colors = plt.cm.RdYlGn((top_20['Mean_Score'] - 1) / 3)  # Scale 1-4 to 0-1
        
        bars = ax.barh(y_pos, top_20['Mean_Score'], color=colors, alpha=0.8)
        
        # Add score labels
        for i, (idx, row) in enumerate(top_20.iterrows()):
            ax.text(row['Mean_Score'] + 0.05, i, f"{row['Mean_Score']:.2f}",
                   va='center', fontweight='bold', fontsize=9)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"{row['Name']} ({row['Location']})" 
                            for _, row in top_20.iterrows()], fontsize=10)
        ax.set_xlabel('Progressiveness Score (1-4 Scale)', fontweight='bold', fontsize=11)
        ax.set_title('Top 20 Most Progressive District Attorneys', 
                    fontweight='bold', fontsize=14, pad=20)
        ax.set_xlim(1, 4)
        ax.axvline(x=2.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'viz_1_rankings.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_viz_all_50(self, output_path):
        """Create comprehensive overview of all 50 DAs"""
        print("   Creating Viz 2: All 50 DAs Overview...")
        
        reliable = self.da_df[self.da_df['Substantive_Ratings'] >= 10].copy()
        
        fig, ax = plt.subplots(figsize=(16, 14))
        
        y_pos = np.arange(len(reliable))
        colors = plt.cm.RdYlGn((reliable['Mean_Score'] - 1) / 3)
        
        bars = ax.barh(y_pos, reliable['Mean_Score'], color=colors, alpha=0.7)
        
        # Add labels with scores and sample sizes
        for i, (idx, row) in enumerate(reliable.iterrows()):
            ax.text(row['Mean_Score'] + 0.05, i, 
                   f"{row['Mean_Score']:.2f} (n={row['Substantive_Ratings']})",
                   va='center', fontsize=8)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"{row['Name']} ({row['Location']})" 
                            for _, row in reliable.iterrows()], fontsize=9)
        ax.set_xlabel('Progressiveness Score (1-4 Scale)', fontweight='bold', fontsize=12)
        ax.set_title('Complete Rankings: 50 Notable District Attorneys', 
                    fontweight='bold', fontsize=16, pad=20)
        ax.set_xlim(1, 4)
        
        # Add reference lines
        ax.axvline(x=2.5, color='gray', linestyle='--', alpha=0.5, linewidth=1, label='Midpoint (2.5)')
        ax.axvline(x=3, color='green', linestyle='--', alpha=0.5, linewidth=1, label='Progressive Threshold (≥3)')
        
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        ax.legend(loc='lower right')
        
        plt.tight_layout()
        plt.savefig(output_path / 'viz_2_all_50_das.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_viz_position_effects(self, output_path):
        """Create position effects visualization"""
        print("   Creating Viz 3: Position Effects...")
        
        positions = ['Academic', 'Defense Attorney', 'Prosecutor']
        
        # Get DAs with all three position ratings
        reliable = self.da_df[
            (self.da_df['Substantive_Ratings'] >= 30) &
            (self.da_df['Academic_Mean'].notna()) &
            (self.da_df['Defense_Mean'].notna()) &
            (self.da_df['Prosecutor_Mean'].notna())
        ].copy()
        
        # Select top 15 for clarity
        top_das = reliable.head(15)
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        x = np.arange(len(top_das))
        width = 0.25
        
        academics = ax.bar(x - width, top_das['Academic_Mean'], width, 
                          label='Academic', color='#3498db', alpha=0.8)
        defense = ax.bar(x, top_das['Defense_Mean'], width, 
                        label='Defense Attorney', color='#e74c3c', alpha=0.8)
        prosecutors = ax.bar(x + width, top_das['Prosecutor_Mean'], width, 
                            label='Prosecutor', color='#2ecc71', alpha=0.8)
        
        ax.set_ylabel('Mean Progressiveness Score', fontweight='bold', fontsize=11)
        ax.set_title('How Different Professionals Rate the Same Prosecutors', 
                    fontweight='bold', fontsize=14, pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels([row['Name'] for _, row in top_das.iterrows()], 
                          rotation=45, ha='right', fontsize=9)
        ax.legend(loc='upper right', fontsize=10)
        ax.set_ylim(1, 4)
        ax.axhline(y=2.5, color='gray', linestyle='--', alpha=0.3)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'viz_3_position_effects.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_viz_familiarity(self, output_path):
        """Create familiarity crisis visualization"""
        print("   Creating Viz 4: Familiarity Crisis...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # Panel 1: Familiarity rates
        reliable = self.da_df[self.da_df['Substantive_Ratings'] >= 10].copy()
        top_20_familiar = reliable.nlargest(20, 'Familiarity_Rate')
        
        y_pos = np.arange(len(top_20_familiar))
        bars = ax1.barh(y_pos, top_20_familiar['Familiarity_Rate'], 
                       color='#3498db', alpha=0.7)
        
        for i, (idx, row) in enumerate(top_20_familiar.iterrows()):
            ax1.text(row['Familiarity_Rate'] + 1, i, f"{row['Familiarity_Rate']:.0f}%",
                    va='center', fontsize=8)
        
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels([row['Name'] for _, row in top_20_familiar.iterrows()], 
                           fontsize=9)
        ax1.set_xlabel('Familiarity Rate (%)', fontweight='bold')
        ax1.set_title('Top 20 Most Familiar Prosecutors', fontweight='bold', fontsize=12)
        ax1.invert_yaxis()
        ax1.grid(axis='x', alpha=0.3)
        
        # Panel 2: Scatter plot - Familiarity vs Mean Score
        reliable_subset = reliable[reliable['Substantive_Ratings'] >= 30]
        
        scatter = ax2.scatter(reliable_subset['Familiarity_Rate'], 
                            reliable_subset['Mean_Score'],
                            c=reliable_subset['Mean_Score'], 
                            cmap='RdYlGn', 
                            s=100, alpha=0.6, 
                            vmin=1, vmax=4)
        
        # Add labels for notable DAs
        for _, row in reliable_subset.head(10).iterrows():
            ax2.annotate(row['Name'], 
                        (row['Familiarity_Rate'], row['Mean_Score']),
                        xytext=(5, 5), textcoords='offset points', 
                        fontsize=7, alpha=0.7)
        
        ax2.set_xlabel('Familiarity Rate (%)', fontweight='bold')
        ax2.set_ylabel('Mean Progressiveness Score', fontweight='bold')
        ax2.set_title('Familiarity vs Progressiveness', fontweight='bold', fontsize=12)
        ax2.grid(alpha=0.3)
        ax2.set_ylim(1, 4)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label('Progressiveness', rotation=270, labelpad=15)
        
        plt.tight_layout()
        plt.savefig(output_path / 'viz_4_familiarity.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_viz_transitions(self, output_path):
        """Create jurisdiction transitions visualization"""
        print("   Creating Viz 5: Electoral Transitions...")
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        transitions_sorted = self.transitions_df.sort_values('Change')
        
        y_pos = np.arange(len(transitions_sorted))
        colors = ['#e74c3c' if x < 0 else '#2ecc71' 
                 for x in transitions_sorted['Change']]
        
        bars = ax.barh(y_pos, transitions_sorted['Change'], color=colors, alpha=0.7)
        
        # Add labels
        for i, (idx, row) in enumerate(transitions_sorted.iterrows()):
            label = f"{row['Predecessor']} → {row['Successor']}\n{row['Jurisdiction']}"
            ax.text(-0.1, i, label, va='center', ha='right', fontsize=9)
            ax.text(row['Change'] + 0.05 if row['Change'] > 0 else row['Change'] - 0.05, 
                   i, f"{row['Change']:+.2f}", 
                   va='center', ha='left' if row['Change'] > 0 else 'right',
                   fontweight='bold', fontsize=9)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels([''] * len(transitions_sorted))
        ax.set_xlabel('Change in Progressiveness Score', fontweight='bold', fontsize=12)
        ax.set_title('Electoral Transitions: How Jurisdictions Shifted', 
                    fontweight='bold', fontsize=14, pad=20)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax.grid(axis='x', alpha=0.3)
        ax.set_xlim(-3, 1)
        
        # Add legend
        traditional_patch = mpatches.Patch(color='#e74c3c', alpha=0.7, label='Shift toward Traditional')
        progressive_patch = mpatches.Patch(color='#2ecc71', alpha=0.7, label='Shift toward Progressive')
        ax.legend(handles=[traditional_patch, progressive_patch], loc='lower right')
        
        plt.tight_layout()
        plt.savefig(output_path / 'viz_5_transitions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_viz_controversy(self, output_path):
        """Create controversy analysis visualization"""
        print("   Creating Viz 6: Controversy Analysis...")
        
        reliable = self.da_df[self.da_df['Substantive_Ratings'] >= 30].copy()
        most_controversial = reliable.nlargest(15, 'Std_Dev')
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        y_pos = np.arange(len(most_controversial))
        
        # Create bars colored by mean score
        colors = plt.cm.RdYlGn((most_controversial['Mean_Score'] - 1) / 3)
        bars = ax.barh(y_pos, most_controversial['Std_Dev'], color=colors, alpha=0.7)
        
        # Add labels
        for i, (idx, row) in enumerate(most_controversial.iterrows()):
            ax.text(row['Std_Dev'] + 0.02, i, 
                   f"σ={row['Std_Dev']:.2f} | μ={row['Mean_Score']:.2f}",
                   va='center', fontsize=9)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"{row['Name']} ({row['Location']})" 
                           for _, row in most_controversial.iterrows()], fontsize=10)
        ax.set_xlabel('Standard Deviation (Disagreement)', fontweight='bold', fontsize=11)
        ax.set_title('Most Controversial Prosecutors: Highest Disagreement in Ratings', 
                    fontweight='bold', fontsize=14, pad=20)
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        
        # Add note
        ax.text(0.95, 0.05, 
               'Bar color indicates mean score\nRed = Traditional, Yellow = Moderate, Green = Progressive',
               transform=ax.transAxes, fontsize=9, va='bottom', ha='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(output_path / 'viz_6_controversy.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_viz_geographic(self, output_path):
        """Create geographic distribution visualization"""
        print("   Creating Viz 7: Geographic Distribution...")
        
        # Get DAs by state
        state_da_counts = {}
        for _, row in self.da_df.iterrows():
            location = row['Location']
            if ', ' in location:
                state = location.split(', ')[-1]
                if state not in state_da_counts:
                    state_da_counts[state] = {
                        'count': 0,
                        'total_score': 0,
                        'das': []
                    }
                state_da_counts[state]['count'] += 1
                if not np.isnan(row['Mean_Score']):
                    state_da_counts[state]['total_score'] += row['Mean_Score']
                    state_da_counts[state]['das'].append(row['Name'])
        
        # Calculate average scores
        state_summary = []
        for state, data in state_da_counts.items():
            if data['count'] > 0:
                avg_score = data['total_score'] / data['count'] if data['total_score'] > 0 else np.nan
                state_summary.append({
                    'state': state,
                    'n_das': data['count'],
                    'avg_score': avg_score
                })
        
        state_df = pd.DataFrame(state_summary).sort_values('n_das', ascending=False).head(15)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Panel 1: Number of DAs per state
        y_pos = np.arange(len(state_df))
        bars1 = ax1.barh(y_pos, state_df['n_das'], color='#3498db', alpha=0.7)
        
        for i, (idx, row) in enumerate(state_df.iterrows()):
            ax1.text(row['n_das'] + 0.1, i, f"{row['n_das']}", 
                    va='center', fontsize=10)
        
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(state_df['state'], fontsize=11)
        ax1.set_xlabel('Number of Prosecutors in National Sample', fontweight='bold')
        ax1.set_title('States with Most Prosecutors in Study', fontweight='bold', fontsize=12)
        ax1.invert_yaxis()
        ax1.grid(axis='x', alpha=0.3)
        
        # Panel 2: Average progressiveness by state
        state_df_scored = state_df[state_df['avg_score'].notna()].sort_values('avg_score', ascending=False)
        
        y_pos2 = np.arange(len(state_df_scored))
        colors = plt.cm.RdYlGn((state_df_scored['avg_score'] - 1) / 3)
        bars2 = ax2.barh(y_pos2, state_df_scored['avg_score'], color=colors, alpha=0.7)
        
        for i, (idx, row) in enumerate(state_df_scored.iterrows()):
            ax2.text(row['avg_score'] + 0.05, i, f"{row['avg_score']:.2f}", 
                    va='center', fontsize=10)
        
        ax2.set_yticks(y_pos2)
        ax2.set_yticklabels(state_df_scored['state'], fontsize=11)
        ax2.set_xlabel('Average Progressiveness Score', fontweight='bold')
        ax2.set_title('Average DA Progressiveness by State', fontweight='bold', fontsize=12)
        ax2.invert_yaxis()
        ax2.grid(axis='x', alpha=0.3)
        ax2.set_xlim(1, 4)
        ax2.axvline(x=2.5, color='gray', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(output_path / 'viz_7_geographic.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_viz_national_vs_state(self, output_path):
        """Create national vs state comparison visualization"""
        print("   Creating Viz 8: National vs State Comparison...")
        
        reliable_national = self.da_df[self.da_df['Substantive_Ratings'] >= self.min_ratings]
        
        national_mean = reliable_national['Mean_Score'].mean()
        state_mean = self.state_df['mean_score'].mean()
        
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # Panel 1: Distribution comparison
        ax1 = fig.add_subplot(gs[0, 0])
        
        ax1.hist(reliable_national['Mean_Score'], bins=20, alpha=0.6, 
                color='#e74c3c', label=f'National DAs (μ={national_mean:.2f})', edgecolor='black')
        
        # Create weighted histogram for state DAs
        state_scores = []
        for _, row in self.state_df.iterrows():
            state_scores.extend([row['mean_score']] * int(row['total_ratings'] / 10))
        
        ax1.hist(state_scores, bins=20, alpha=0.6, 
                color='#3498db', label=f'State DAs (μ={state_mean:.2f})', edgecolor='black')
        
        ax1.set_xlabel('Progressiveness Score', fontweight='bold')
        ax1.set_ylabel('Frequency', fontweight='bold')
        ax1.set_title('Score Distribution: National vs State', fontweight='bold', fontsize=13)
        ax1.legend()
        ax1.grid(alpha=0.3)
        ax1.set_xlim(1, 4)
        
        # Panel 2: Ideological camps
        ax2 = fig.add_subplot(gs[0, 1])
        
        # National camps
        nat_prog = len(reliable_national[reliable_national['Mean_Score'] >= 3])
        nat_trad = len(reliable_national[reliable_national['Mean_Score'] <= 2])
        nat_mod = len(reliable_national[(reliable_national['Mean_Score'] > 2) & 
                                        (reliable_national['Mean_Score'] < 3)])
        
        # State camps
        state_prog = len(self.state_df[self.state_df['mean_score'] >= 3])
        state_trad = len(self.state_df[self.state_df['mean_score'] <= 2])
        state_mod = len(self.state_df[(self.state_df['mean_score'] > 2) & 
                                      (self.state_df['mean_score'] < 3)])
        
        x = np.arange(3)
        width = 0.35
        
        nat_bars = ax2.bar(x - width/2, 
                          [nat_trad/len(reliable_national)*100,
                           nat_mod/len(reliable_national)*100,
                           nat_prog/len(reliable_national)*100],
                          width, label='National DAs', color='#e74c3c', alpha=0.7)
        
        state_bars = ax2.bar(x + width/2,
                            [state_trad/len(self.state_df)*100,
                             state_mod/len(self.state_df)*100,
                             state_prog/len(self.state_df)*100],
                            width, label='State DAs', color='#3498db', alpha=0.7)
        
        ax2.set_ylabel('Percentage', fontweight='bold')
        ax2.set_title('Ideological Distribution', fontweight='bold', fontsize=13)
        ax2.set_xticks(x)
        ax2.set_xticklabels(['Traditional\n(≤2)', 'Moderate\n(>2, <3)', 'Progressive\n(≥3)'])
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bars in [nat_bars, state_bars]:
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.0f}%', ha='center', va='bottom', fontsize=9)
        
        # Panel 3: Top DAs from each category
        ax3 = fig.add_subplot(gs[1, 0])
        
        top_national = reliable_national.nlargest(10, 'Mean_Score')
        y_pos = np.arange(len(top_national))
        
        colors = plt.cm.RdYlGn((top_national['Mean_Score'] - 1) / 3)
        bars = ax3.barh(y_pos, top_national['Mean_Score'], color=colors, alpha=0.7)
        
        for i, (idx, row) in enumerate(top_national.iterrows()):
            ax3.text(row['Mean_Score'] + 0.05, i, f"{row['Mean_Score']:.2f}", 
                    va='center', fontsize=8)
        
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels([f"{row['Name']}" for _, row in top_national.iterrows()], fontsize=9)
        ax3.set_xlabel('Progressiveness Score', fontweight='bold')
        ax3.set_title('Top 10 National DAs', fontweight='bold', fontsize=13)
        ax3.invert_yaxis()
        ax3.grid(axis='x', alpha=0.3)
        ax3.set_xlim(1, 4)
        
        # Panel 4: Summary statistics
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis('off')
        
        gap = national_mean - state_mean
        gap_pct = (gap / 3) * 100  # 3 is the range of the scale
        
        prog_count = nat_prog
        mod_count = nat_mod
        trad_count = nat_trad
        
        most_known = reliable_national.nlargest(1, 'Familiarity_Rate')
        most_known_name = most_known['Name'].values[0]
        most_known_pct = most_known['Familiarity_Rate'].values[0]
        
        avg_unfamiliarity = 100 - reliable_national['Familiarity_Rate'].mean()
        
        summary_text = f"""
**KEY FINDINGS & OBSERVATIONS:**

**1. National vs. State Gap**
   • **National DAs:** Rated at **{national_mean:.2f}** (Progressive-Leaning)
   • **State-Level DAs:** Rated at **{state_mean:.2f}** (Traditional-Leaning)
   • **Gap:** A significant **{gap:.2f} point** difference,
     representing **{gap_pct:.0f}%** of the rating scale.

**2. National DA Ideological Camps**
   • **Progressive (≥3):** {prog_count} DAs ({prog_count/len(reliable_national):.0%})
   • **Moderate (>2 and <3):** {mod_count} DAs ({mod_count/len(reliable_national):.0%})
   • **Traditional (≤2):** {trad_count} DAs ({trad_count/len(reliable_national):.0%})

**3. The Familiarity Factor**
   • The public perception of prosecutors appears
     heavily influenced by a few famous figures.
   • **Average Unfamiliarity** with national DAs is **{avg_unfamiliarity:.0f}%.**
   • **Most Known:** {most_known_name} is still only
     recognized by **{most_known_pct:.0f}%** of respondents.
"""
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0f0f0', alpha=0.8))
        
        plt.suptitle('The Two-Tier System of Prosecutor Perception: National vs. State',
                    fontsize=16, fontweight='bold', y=0.98)
        
        plt.savefig(output_path / 'viz_8_national_vs_state.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # =========================================================================
    # PART 10: EXPORT AND REPORTING
    # =========================================================================
    
    def export_results(self, output_dir='output'):
        """Export all analysis results to CSV files"""
        print("=" * 80)
        print("PART 10: EXPORTING RESULTS")
        print("=" * 80)
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Export main DA analysis
        export_df = self.da_df[['Name', 'Location', 'Mean_Score', 'Median_Score', 'Std_Dev',
                                'Substantive_Ratings', 'Familiarity_Rate', 
                                'Very_Traditional', 'Traditional', 'Progressive', 'Very_Progressive',
                                'Academic_Mean', 'Defense_Mean', 'Prosecutor_Mean']].copy()
        export_df = export_df.sort_values('Mean_Score', ascending=False)
        export_df['Rank'] = range(1, len(export_df) + 1)
        export_df = export_df[['Rank'] + [col for col in export_df.columns if col != 'Rank']]
        export_df.to_csv(output_path / 'notable_prosecutors_complete.csv', index=False)
        print("   ✓ Exported: notable_prosecutors_complete.csv")
        
        # Export transitions
        if hasattr(self, 'transitions_df'):
            self.transitions_df.to_csv(output_path / 'jurisdiction_transitions.csv', index=False)
            print("   ✓ Exported: jurisdiction_transitions.csv")
        
        # Export state analysis
        if hasattr(self, 'state_df') and len(self.state_df) > 0:
            self.state_df.to_csv(output_path / 'state_level_analysis.csv', index=False)
            print("   ✓ Exported: state_level_analysis.csv")
        
        # Export position analysis
        if hasattr(self, 'position_stats'):
            self.position_stats.to_csv(output_path / 'position_analysis.csv', index=False)
            print("   ✓ Exported: position_analysis.csv")
        
        # Export filtered data
        self.df_consented.to_csv(output_path / 'filtered_survey_data.csv', index=False)
        print("   ✓ Exported: filtered_survey_data.csv")
        
        print(f"\n   All results saved to: {output_dir}/")
        print()
    
    def generate_final_report(self, output_dir='output'):
        """Generate comprehensive final report"""
        print("=" * 80)
        print("GENERATING FINAL COMPREHENSIVE REPORT")
        print("=" * 80)
        
        output_path = Path(output_dir)
        
        # Calculate statistics
        reliable = self.da_df[self.da_df['Substantive_Ratings'] >= self.min_ratings]
        progressive_count = len(reliable[reliable['Mean_Score'] >= 3])
        traditional_count = len(reliable[reliable['Mean_Score'] <= 2])
        moderate_count = len(reliable[(reliable['Mean_Score'] > 2) & (reliable['Mean_Score'] < 3)])
        
        # Position breakdown
        positions = ['Academic (e.g., law professor, criminal justice researcher)',
                    'Defense Attorney', 'Prosecutor']
        
        report = f"""
{'=' * 80}
COMPREHENSIVE ANALYSIS REPORT: DA PROGRESSIVENESS SURVEY
{'=' * 80}

ANALYSIS METADATA:
- Analysis Date: October 2025
- Minimum Ratings Threshold: {self.min_ratings}
- Scale: 1-4 (Very Traditional=1, Traditional=2, Progressive=3, Very Progressive=4)
- Not Familiar responses excluded from mean calculations

{'=' * 80}
PART 1: SAMPLE COMPOSITION
{'=' * 80}

Total Respondents (Consented): {len(self.df_consented)}
"""
        
        for position in positions:
            count = len(self.df_consented[self.df_consented['position'] == position])
            pct = count / len(self.df_consented) * 100
            report += f"  - {position}: {count} ({pct:.1f}%)\n"
        
        report += f"""
{'=' * 80}
PART 2: FAMILIARITY ANALYSIS
{'=' * 80}

Notable Prosecutors (50 National DAs):
  - Total rating opportunities: {self.familiarity_metrics['total_opportunities']:,}
  - Explicit "Not Familiar": {self.familiarity_metrics['explicit_not_familiar']:,} ({self.familiarity_metrics['explicit_not_familiar']/self.familiarity_metrics['total_opportunities']*100:.1f}%)
  - Left blank: {self.familiarity_metrics['left_blank']:,} ({self.familiarity_metrics['left_blank']/self.familiarity_metrics['total_opportunities']*100:.1f}%)
  - Substantive ratings: {self.familiarity_metrics['substantive']:,} ({self.familiarity_metrics['substantive']/self.familiarity_metrics['total_opportunities']*100:.1f}%)

Average familiarity rate: {reliable['Familiarity_Rate'].mean():.1f}%
Most familiar: {reliable.nlargest(1, 'Familiarity_Rate')['Name'].values[0]} ({reliable.nlargest(1, 'Familiarity_Rate')['Familiarity_Rate'].values[0]:.1f}%)

{'=' * 80}
PART 3: OVERALL PROGRESSIVENESS DISTRIBUTION
{'=' * 80}

Prosecutors with ≥{self.min_ratings} ratings: {len(reliable)}

Mean score for all 50 national DAs: {reliable['Mean_Score'].mean():.2f}
Median score: {reliable['Mean_Score'].median():.2f}

Ideological Distribution:
  - Progressive (≥3): {progressive_count} ({progressive_count/len(reliable)*100:.0f}%)
  - Moderate (>2 and <3): {moderate_count} ({moderate_count/len(reliable)*100:.0f}%)
  - Traditional (≤2): {traditional_count} ({traditional_count/len(reliable)*100:.0f}%)

{'=' * 80}
PART 4: TOP 5 MOST PROGRESSIVE DAs
{'=' * 80}
"""
        
        for i, (idx, row) in enumerate(reliable.head(5).iterrows(), 1):
            report += f"\n{i}. {row['Name']} ({row['Location']})\n"
            report += f"   Mean Score: {row['Mean_Score']:.2f}\n"
            report += f"   N Ratings: {row['Substantive_Ratings']}\n"
            report += f"   Familiarity: {row['Familiarity_Rate']:.0f}%\n"
            report += f"   Very Progressive: {row['Very_Progressive_Pct']:.0f}%\n"
        
        report += f"""
{'=' * 80}
PART 5: TOP 5 MOST TRADITIONAL DAs
{'=' * 80}
"""
        
        for i, (idx, row) in enumerate(reliable.tail(5).sort_values('Mean_Score').iterrows(), 1):
            report += f"\n{i}. {row['Name']} ({row['Location']})\n"
            report += f"   Mean Score: {row['Mean_Score']:.2f}\n"
            report += f"   N Ratings: {row['Substantive_Ratings']}\n"
            report += f"   Familiarity: {row['Familiarity_Rate']:.0f}%\n"
            report += f"   Very Traditional: {row['Very_Traditional_Pct']:.0f}%\n"
        
        # Add transition analysis if available
        if hasattr(self, 'transitions_df'):
            report += f"""
{'=' * 80}
PART 6: ELECTORAL TRANSITIONS
{'=' * 80}

Tracked Jurisdictions: {len(self.transitions_df)}
Shifts toward traditional: {len(self.transitions_df[self.transitions_df['Change'] < 0])}
Shifts toward progressive: {len(self.transitions_df[self.transitions_df['Change'] > 0])}
Average change: {self.transitions_df['Change'].mean():+.2f}

Largest Shifts:
"""
            for i, (idx, row) in enumerate(self.transitions_df.head(3).iterrows(), 1):
                report += f"\n{i}. {row['Jurisdiction']}\n"
                report += f"   {row['Predecessor']} ({row['Pred_Score']:.2f}) → {row['Successor']} ({row['Succ_Score']:.2f})\n"
                report += f"   Change: {row['Change']:+.2f}\n"
        
        # Add state analysis if available
        if hasattr(self, 'state_df') and len(self.state_df) > 0:
            national_mean = reliable['Mean_Score'].mean()
            state_mean = self.state_df['mean_score'].mean()
            
            report += f"""
{'=' * 80}
PART 7: NATIONAL VS STATE COMPARISON
{'=' * 80}

National DAs Mean: {national_mean:.2f}
State DAs Mean: {state_mean:.2f}
Gap: {national_mean - state_mean:.2f} points

This represents a significant divergence in how the most prominent
prosecutors are perceived compared to the broader prosecutorial landscape.
"""
        
        report += f"""
{'=' * 80}
PART 8: POSITION EFFECTS
{'=' * 80}

Key Finding: Different professional groups show distinct rating patterns:
"""
        
        if hasattr(self, 'position_stats'):
            for _, row in self.position_stats.iterrows():
                report += f"\n{row['position']}:\n"
                report += f"   Mean Score: {row['mean_score']:.2f}\n"
                report += f"   % Progressive: {row['progressive_pct']:.1f}%\n"
                report += f"   % Traditional: {row['traditional_pct']:.1f}%\n"
        
        report += f"""
{'=' * 80}
SUMMARY AND IMPLICATIONS
{'=' * 80}

1. FAMILIARITY CRISIS: The vast majority of prosecutors, even at the state
   level, are unfamiliar to criminal justice professionals.

2. PROGRESSIVE BIAS IN NATIONAL SAMPLE: The 50 most notable prosecutors
   skew significantly more progressive than state-level DAs.

3. ELECTORAL TREND: Recent transitions show a movement toward more
   traditional prosecutors in multiple jurisdictions.

4. POSITION EFFECTS: Prosecutors, defense attorneys, and academics rate
   the same DAs differently, suggesting varying perspectives on what
   constitutes "progressive" prosecution.

5. CONTROVERSY: High standard deviations for certain DAs indicate deep
   disagreement about their classification.

{'=' * 80}
ANALYSIS COMPLETE
{'=' * 80}

All outputs have been saved to the output directory.
For questions or issues, contact: Dvir Yogev, BERQ-J
"""
        
        # Save report
        with open(output_path / 'final_comprehensive_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(report)
        print("\n✅ Comprehensive report saved to: final_comprehensive_report.txt")
        print()
    
    def run_complete_analysis(self, output_dir='output'):
        """Run the complete analysis pipeline"""
        print("\n" + "=" * 80)
        print("STARTING COMPREHENSIVE PROSECUTOR IDEOLOGY ANALYSIS")
        print("=" * 80)
        print()
        
        # Run all analyses
        self.analyze_familiarity()
        self.analyze_notable_prosecutors()
        self.analyze_jurisdiction_transitions()
        self.analyze_state_patterns()
        self.analyze_by_position()
        self.analyze_open_text()
        self.analyze_geographic_distribution()
        self.analyze_additional_metrics()
        
        # Create visualizations
        self.create_all_visualizations(output_dir)
        
        # Export results
        self.export_results(output_dir)
        
        # Generate final report
        self.generate_final_report(output_dir)
        
        print("=" * 80)
        print("✅ COMPLETE ANALYSIS FINISHED SUCCESSFULLY!")
        print("=" * 80)
        print(f"\nAll outputs saved to: {output_dir}/")
        print("\nGenerated files:")
        print("  📊 Data Files:")
        print("     - notable_prosecutors_complete.csv")
        print("     - jurisdiction_transitions.csv")
        print("     - state_level_analysis.csv")
        print("     - position_analysis.csv")
        print("     - filtered_survey_data.csv")
        print("\n  📈 Visualizations:")
        print("     - viz_1_rankings.png")
        print("     - viz_2_all_50_das.png")
        print("     - viz_3_position_effects.png")
        print("     - viz_4_familiarity.png")
        print("     - viz_5_transitions.png")
        print("     - viz_6_controversy.png")
        print("     - viz_7_geographic.png")
        print("     - viz_8_national_vs_state.png")
        print("\n  📄 Reports:")
        print("     - final_comprehensive_report.txt")
        print("\n" + "=" * 80)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function"""
    
    # Configuration
    survey_file = 'PP_survey_draft_October_16__2025_13_32.csv'
    output_dir = 'output'
    min_ratings_threshold = 10
    
    print("\n" + "=" * 80)
    print("COMPREHENSIVE PROSECUTOR IDEOLOGY SURVEY ANALYSIS")
    print("=" * 80)
    print(f"\n📁 Survey file: {survey_file}")
    print(f"📁 Output directory: {output_dir}")
    print(f"📊 Minimum ratings threshold: {min_ratings_threshold}")
    print(f"📏 Scale: 1-4 (Very Traditional=1, Traditional=2, Progressive=3, Very Progressive=4)")
    print("=" * 80)
    
    # Check if file exists
    if not Path(survey_file).exists():
        print(f"\n❌ ERROR: Survey file not found: {survey_file}")
        print("Please ensure the file is in the current directory.")
        return
    
    # Initialize analyzer
    analyzer = ComprehensiveProsecutorAnalyzer(
        survey_file=survey_file,
        min_ratings_threshold=min_ratings_threshold
    )
    
    # Run complete analysis
    analyzer.run_complete_analysis(output_dir=output_dir)


if __name__ == '__main__':
    main()

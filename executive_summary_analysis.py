"""
Prosecutor Ideology Survey - Executive Summary Analysis Script
===============================================================
This script generates all key statistics and findings for the executive summary,
including:
- Notable prosecutor rankings (most progressive/traditional)
- Jurisdiction transition analysis
- State-level patterns
- Respondent analysis by position
- Open-text response themes
- Familiarity patterns

Author: Dvir Yogev, BERQ-J
Date: October 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class ExecutiveSummaryAnalyzer:
    """Analyzer for generating executive summary statistics"""
    
    def __init__(self, survey_file, min_ratings_threshold=10):
        """
        Initialize the analyzer
        
        Parameters:
        -----------
        survey_file : str
            Path to the Qualtrics survey CSV file
        min_ratings_threshold : int
            Minimum number of ratings needed to include a prosecutor
        """
        self.survey_file = survey_file
        self.min_ratings = min_ratings_threshold
        
        # Rating scale mapping
        self.rating_map = {
            'Very Traditional': 1,
            'Traditional': 2,
            'Progressive': 3,
            'Very Progressive': 4,
            'Not Familiar': np.nan
        }
        
        self.load_data()
    
    def load_data(self):
        """Load and prepare the survey data"""
        print("="*80)
        print("LOADING DATA")
        print("="*80)
        
        # Load survey data (skip row 1 which contains question text)
        self.df = pd.read_csv(self.survey_file, skiprows=[1])
        self.questions_df = pd.read_csv(self.survey_file, nrows=1)
        
        print(f"✓ Loaded {len(self.df)} survey responses")
        print(f"✓ Completed responses: {self.df['Finished'].sum()}")
        
        # Identify notable prosecutor columns
        self.notable_cols = [col for col in self.df.columns if col.startswith('notable_')]
        print(f"✓ Identified {len(self.notable_cols)} notable prosecutors")
        
        # Identify state DA columns
        self.state_da_cols = []
        for col in self.df.columns:
            if '_' in col and not col.startswith('Q') and not col.startswith('notable'):
                if col not in ['gender_4_TEXT', 'race_1', 'race_2', 'race_3', 'race_4', 
                              'race_5', 'race_6', 'race_7', 'Q55_5_TEXT']:
                    unique_vals = self.df[col].dropna().unique()
                    if len(unique_vals) > 0 and any('Progressive' in str(v) or 'Traditional' in str(v) 
                                                     for v in unique_vals):
                        self.state_da_cols.append(col)
        
        print(f"✓ Identified {len(self.state_da_cols)} state prosecutor columns")
    
    def get_prosecutor_name(self, col):
        """Extract prosecutor name and location from question text"""
        if col in self.questions_df.columns:
            question_text = self.questions_df[col].iloc[0]
            if ' - ' in question_text:
                return question_text.split(' - ')[-1].strip()
        return 'Unknown'
    
    def analyze_familiarity(self):
        """Calculate familiarity statistics"""
        print("\n" + "="*80)
        print("FAMILIARITY ANALYSIS")
        print("="*80)
        
        # Notable prosecutors familiarity
        rated_at_least_one = self.df[self.notable_cols].notna().any(axis=1)
        respondents_who_rated = rated_at_least_one.sum()
        
        total_opportunities = respondents_who_rated * len(self.notable_cols)
        explicit_not_familiar = 0
        left_blank = 0
        substantive = 0
        
        for idx, row in self.df[rated_at_least_one].iterrows():
            for col in self.notable_cols:
                val = row[col]
                if pd.isna(val):
                    left_blank += 1
                elif val == 'Not Familiar':
                    explicit_not_familiar += 1
                else:
                    substantive += 1
        
        total_unfamiliar = explicit_not_familiar + left_blank
        
        print(f"\nNOTABLE PROSECUTORS (National Sample):")
        print(f"  Respondents who rated: {respondents_who_rated}")
        print(f"  Total opportunities: {total_opportunities:,}")
        print(f"  Explicit 'Not Familiar': {explicit_not_familiar:,} ({explicit_not_familiar/total_opportunities*100:.1f}%)")
        print(f"  Left blank: {left_blank:,} ({left_blank/total_opportunities*100:.1f}%)")
        print(f"  Total unfamiliar: {total_unfamiliar:,} ({total_unfamiliar/total_opportunities*100:.1f}%)")
        print(f"  Substantive ratings: {substantive:,} ({substantive/total_opportunities*100:.1f}%)")
        
        # Completion pattern
        ratings_per_respondent = []
        for idx, row in self.df[rated_at_least_one].iterrows():
            count = sum(1 for col in self.notable_cols if pd.notna(row[col]))
            ratings_per_respondent.append(count)
        
        rated_all_50 = sum(1 for x in ratings_per_respondent if x == 50)
        print(f"\n  Completed all 50 prosecutors: {rated_all_50} ({rated_all_50/respondents_who_rated*100:.1f}%)")
        
        # State prosecutors familiarity
        agreed_to_rate_state = self.df['more'] == 'Show me the prosecutors from my state'
        
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
        
        for idx, row in self.df[agreed_to_rate_state].iterrows():
            respondent_state = row['Q1']
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
        
        state_total_unfam = state_explicit_nf + state_blank
        
        print(f"\nSTATE PROSECUTORS (Within-State Sample):")
        print(f"  Agreed to rate state DAs: {agreed_to_rate_state.sum()}")
        print(f"  Total opportunities: {state_total_opp:,}")
        print(f"  Explicit 'Not Familiar': {state_explicit_nf:,} ({state_explicit_nf/state_total_opp*100:.1f}%)")
        print(f"  Left blank: {state_blank:,} ({state_blank/state_total_opp*100:.1f}%)")
        print(f"  Total unfamiliar: {state_total_unfam:,} ({state_total_unfam/state_total_opp*100:.1f}%)")
        print(f"  Substantive ratings: {state_substantive:,} ({state_substantive/state_total_opp*100:.1f}%)")
    
    def analyze_notable_prosecutors(self):
        """Analyze notable prosecutor ratings"""
        print("\n" + "="*80)
        print("NOTABLE PROSECUTOR RANKINGS")
        print("="*80)
        
        results = []
        for col in self.notable_cols:
            responses = self.df[col].dropna()
            responses = responses[responses != 'Not Familiar']
            
            if len(responses) >= self.min_ratings:
                numeric = responses.map(self.rating_map)
                
                results.append({
                    'column': col,
                    'name_location': self.get_prosecutor_name(col),
                    'n_ratings': len(numeric),
                    'mean_score': numeric.mean(),
                    'median_score': numeric.median(),
                    'std_score': numeric.std(),
                    'very_progressive_pct': (numeric == 4).sum() / len(numeric) * 100,
                    'progressive_pct': (numeric == 3).sum() / len(numeric) * 100,
                    'traditional_pct': (numeric == 2).sum() / len(numeric) * 100,
                    'very_traditional_pct': (numeric == 1).sum() / len(numeric) * 100
                })
        
        self.notable_results = pd.DataFrame(results).sort_values('mean_score', ascending=False)
        
        print(f"\nProsecutors with >= {self.min_ratings} ratings: {len(self.notable_results)}\n")
        
        print("TOP 10 MOST PROGRESSIVE:")
        for idx, (i, row) in enumerate(self.notable_results.head(10).iterrows(), 1):
            print(f"  {idx}. {row['name_location']}")
            print(f"     Mean: {row['mean_score']:.2f} | N={row['n_ratings']} | "
                  f"Very Prog: {row['very_progressive_pct']:.0f}%")
        
        print("\nTOP 10 MOST TRADITIONAL:")
        for idx, (i, row) in enumerate(self.notable_results.tail(10).sort_values('mean_score').iterrows(), 1):
            print(f"  {idx}. {row['name_location']}")
            print(f"     Mean: {row['mean_score']:.2f} | N={row['n_ratings']} | "
                  f"Very Trad: {row['very_traditional_pct']:.0f}%")
        
        return self.notable_results
    
    def analyze_jurisdiction_transitions(self):
        """Analyze jurisdictions that experienced transitions"""
        print("\n" + "="*80)
        print("JURISDICTION TRANSITIONS")
        print("="*80)
        
        # Define known transitions
        transitions = [
            {
                'jurisdiction': 'San Francisco, CA',
                'direction': 'Progressive → Traditional (Recall 2022)',
                'progressive': ('Chesa Boudin', 'notable_5'),
                'traditional': ('Brooke Jenkins', 'notable_4')
            },
            {
                'jurisdiction': 'Los Angeles, CA',
                'direction': 'Progressive → Traditional (Election 2024)',
                'progressive': ('George Gascón', 'notable_8'),
                'traditional': ('Nathan Hochman', 'notable_9')
            },
            {
                'jurisdiction': 'Multnomah County (Portland), OR',
                'direction': 'Progressive → Traditional (Election 2024)',
                'progressive': ('Mike Schmidt', 'notable_37'),
                'traditional': ('Nathan Vasquez', 'notable_38')
            },
            {
                'jurisdiction': 'Alameda County, CA',
                'direction': 'Traditional → Progressive (Election 2022)',
                'traditional': ('Nancy O\'Malley', 'notable_7'),
                'progressive': ('Pamela Price', 'notable_6')
            }
        ]
        
        for trans in transitions:
            print(f"\n{trans['jurisdiction']} - {trans['direction']}")
            
            prog_name, prog_col = trans['progressive']
            trad_name, trad_col = trans['traditional']
            
            prog_ratings = self.df[prog_col].map(self.rating_map).dropna()
            trad_ratings = self.df[trad_col].map(self.rating_map).dropna()
            
            prog_total = self.df[prog_col].notna().sum()
            trad_total = self.df[trad_col].notna().sum()
            
            print(f"  {prog_name}: Mean={prog_ratings.mean():.2f}, N={len(prog_ratings)}, "
                  f"Familiarity={len(prog_ratings)/prog_total*100:.0f}%")
            print(f"  {trad_name}: Mean={trad_ratings.mean():.2f}, N={len(trad_ratings)}, "
                  f"Familiarity={len(trad_ratings)/trad_total*100:.0f}%")
            print(f"  Ideological shift: {abs(prog_ratings.mean() - trad_ratings.mean()):.2f} points")
            print(f"  Familiarity change: "
                  f"{(len(trad_ratings)/trad_total - len(prog_ratings)/prog_total)*100:+.0f} percentage points")
    
    def analyze_state_patterns(self):
        """Analyze state-level prosecutor patterns"""
        print("\n" + "="*80)
        print("STATE-LEVEL PATTERNS")
        print("="*80)
        
        state_results = {}
        for col in self.state_da_cols:
            state = col.rsplit('_', 1)[0]
            ratings = self.df[col].map(self.rating_map).dropna()
            
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
        
        print(f"\nStates with sufficient data (≥10 ratings): {len(self.state_df)}\n")
        
        print("TOP 5 MOST PROGRESSIVE STATES:")
        for idx, (i, row) in enumerate(self.state_df.head(5).iterrows(), 1):
            print(f"  {idx}. {row['state']}: Mean={row['mean_score']:.2f}, "
                  f"N={row['total_ratings']}, Progressive={row['progressive_pct']:.0f}%")
        
        print("\nTOP 5 MOST TRADITIONAL STATES:")
        for idx, (i, row) in enumerate(self.state_df.tail(5).sort_values('mean_score').iterrows(), 1):
            print(f"  {idx}. {row['state']}: Mean={row['mean_score']:.2f}, "
                  f"N={row['total_ratings']}, Progressive={row['progressive_pct']:.0f}%")
        
        return self.state_df
    
    def analyze_by_position(self):
        """Analyze rating patterns by respondent position"""
        print("\n" + "="*80)
        print("ANALYSIS BY PROFESSIONAL POSITION")
        print("="*80)
        
        position_counts = self.df['position'].value_counts()
        print("\nSample Composition:")
        for pos, count in position_counts.head(7).items():
            print(f"  {pos}: {count} ({count/len(self.df)*100:.1f}%)")
        
        main_positions = ['Prosecutor', 'Defense Attorney', 
                          'Academic (e.g., law professor, criminal justice researcher)']
        
        print("\nRating Patterns by Position:")
        for position in main_positions:
            subset = self.df[self.df['position'] == position]
            
            all_ratings = []
            for col in self.notable_cols:
                ratings = subset[col].map(self.rating_map).dropna()
                all_ratings.extend(ratings.tolist())
            
            if len(all_ratings) > 0:
                ratings_array = np.array(all_ratings)
                print(f"\n{position}:")
                print(f"  N respondents: {len(subset)}")
                print(f"  Total ratings: {len(all_ratings)}")
                print(f"  Mean score: {ratings_array.mean():.2f}")
                print(f"  % Progressive/Very Progressive: "
                      f"{(ratings_array >= 3).sum()/len(ratings_array)*100:.1f}%")
                print(f"  % Traditional/Very Traditional: "
                      f"{(ratings_array <= 2).sum()/len(ratings_array)*100:.1f}%")
    
    def analyze_open_text(self):
        """Analyze open-text responses"""
        print("\n" + "="*80)
        print("OPEN-TEXT RESPONSE ANALYSIS")
        print("="*80)
        
        q54_responses = self.df['Q54'].dropna()
        print(f"\nQ54 (Progressive characteristics): {len(q54_responses)} responses")
        
        # Theme analysis
        themes = {
            'Rehabilitation/Treatment': ['rehab', 'treatment', 'diversion'],
            'Reduce Incarceration': ['reduce', 'decreas', 'incarceration', 'mass incarceration'],
            'Bail Reform': ['bail', 'pretrial'],
            'Racial Justice': ['racial', 'race', 'equity', 'disparit'],
            'Police Accountability': ['police', 'misconduct', 'officer'],
            'Transparency': ['transparent', 'accountab', 'data']
        }
        
        print(f"\nTheme Frequency:")
        for theme, keywords in themes.items():
            count = sum(1 for r in q54_responses if any(kw in str(r).lower() for kw in keywords))
            pct = count / len(q54_responses) * 100
            print(f"  {theme}: {count} ({pct:.0f}%)")
        
        # Impact assessment
        q55_responses = self.df['Q55'].value_counts()
        print(f"\nQ55 (Impact on reform movement):")
        for val, count in q55_responses.items():
            pct = count / q55_responses.sum() * 100
            print(f"  {val}: {count} ({pct:.1f}%)")
    
    def analyze_geographic_distribution(self):
        """Analyze geographic distribution of respondents"""
        print("\n" + "="*80)
        print("GEOGRAPHIC DISTRIBUTION")
        print("="*80)
        
        if 'Q1' in self.df.columns:
            state_counts = self.df['Q1'].value_counts().head(10)
            print("\nTop 10 states by respondent location:")
            for state, count in state_counts.items():
                print(f"  {state}: {count}")
    
    def export_results(self, output_dir='output'):
        """Export all results to CSV files"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print("\n" + "="*80)
        print("EXPORTING RESULTS")
        print("="*80)
        
        if hasattr(self, 'notable_results'):
            self.notable_results.to_csv(output_path / 'notable_prosecutors_rankings.csv', index=False)
            print(f"✓ Exported notable prosecutor rankings")
        
        if hasattr(self, 'state_df'):
            self.state_df.to_csv(output_path / 'state_level_rankings.csv', index=False)
            print(f"✓ Exported state-level rankings")
        
        print(f"\nAll results saved to: {output_dir}/")
    
    def run_full_analysis(self, output_dir='output'):
        """Run complete analysis pipeline"""
        print("\n" + "="*80)
        print("PROSECUTOR IDEOLOGY SURVEY - EXECUTIVE SUMMARY ANALYSIS")
        print("="*80)
        
        self.analyze_familiarity()
        self.analyze_notable_prosecutors()
        self.analyze_jurisdiction_transitions()
        self.analyze_state_patterns()
        self.analyze_by_position()
        self.analyze_open_text()
        self.analyze_geographic_distribution()
        self.export_results(output_dir)
        
        print("\n" + "="*80)
        print("✅ ANALYSIS COMPLETE!")
        print("="*80)


def main():
    """Main execution function"""
    # Configuration
    survey_file = 'PP_survey_draft_October_16__2025_13_32.csv'
    output_dir = 'output'
    min_ratings_threshold = 10
    
    print("\n" + "="*80)
    print("PROSECUTOR IDEOLOGY SURVEY - EXECUTIVE SUMMARY ANALYSIS")
    print("="*80)
    print(f"Survey file: {survey_file}")
    print(f"Output directory: {output_dir}")
    print(f"Minimum ratings threshold: {min_ratings_threshold}")
    print("="*80)
    
    if not Path(survey_file).exists():
        print(f"\n❌ ERROR: Survey file not found: {survey_file}")
        print("Please update the 'survey_file' path to point to your CSV file.")
        return
    
    # Run analysis
    analyzer = ExecutiveSummaryAnalyzer(
        survey_file=survey_file,
        min_ratings_threshold=min_ratings_threshold
    )
    
    analyzer.run_full_analysis(output_dir=output_dir)


if __name__ == '__main__':
    main()

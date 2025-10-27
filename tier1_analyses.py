#!/usr/bin/env python3
"""
TIER 1 RESEARCH QUESTIONS - IMMEDIATE ANALYSIS
================================================
Running 5 research questions that can be answered with existing merged data

Author: Dvir Yogev, BERQ-J
Date: October 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

# Set up plotting
plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_style("whitegrid")

# Load the merged data
notable_df = pd.read_csv('output/notable_prosecutors_elections.csv')
state_df = pd.read_csv('output/state_prosecutors_elections.csv')

# Fix the years column (it's stored as string)
notable_df['years'] = notable_df['years'].apply(lambda x: eval(x) if isinstance(x, str) and x.startswith('[') else [])
state_df['years'] = state_df['years'].apply(lambda x: eval(x) if isinstance(x, str) and x.startswith('[') else [])

print("="*80)
print("TIER 1 RESEARCH QUESTIONS - IMMEDIATE ANALYSIS")
print("="*80)
print()

# =============================================================================
# RESEARCH QUESTION 1: Ideology × Electoral Competition
# =============================================================================
print("="*80)
print("RQ1: DO PROGRESSIVE PROSECUTORS FACE MORE CONTESTED ELECTIONS?")
print("="*80)
print()

# Notable prosecutors
notable_with_elections = notable_df[notable_df['num_elections'] > 0].copy()

# Calculate ideology scores (we have mean_score which is the familiarity rate, need to get from names)
# Actually, we need to load the original analysis to get mean ideology scores
# Let me recalculate from the survey data

# For now, let's categorize based on known progressive/traditional prosecutors
progressive_names = ['Larry Krasner', 'Chesa Boudin', 'George Gascon', 'Pamela Price', 
                     'Mary Moriarty', 'Kim Foxx', 'Kim Gardner', 'Jose Garza']
traditional_names = ['Rachel Mitchell', 'Amy Weirich', 'Brooke Jenkins', 'Nathan Hochman',
                    'Summer Stephan', 'Katherine Fernandez-Rundle']

notable_with_elections['progressive'] = notable_with_elections['name'].isin(progressive_names)
notable_with_elections['traditional'] = notable_with_elections['name'].isin(traditional_names)

# Compare contestation rates
print("NOTABLE PROSECUTORS:")
print()

prog = notable_with_elections[notable_with_elections['progressive']]
trad = notable_with_elections[notable_with_elections['traditional']]

print(f"Progressive prosecutors (n={len(prog)}):")
print(f"  Ever contested primary: {prog['ever_contested_primary'].mean()*100:.1f}%")
print(f"  Ever contested general: {prog['ever_contested_general'].mean()*100:.1f}%")
print(f"  Ever contested any: {prog['ever_contested_any'].mean()*100:.1f}%")
print()

print(f"Traditional prosecutors (n={len(trad)}):")
print(f"  Ever contested primary: {trad['ever_contested_primary'].mean()*100:.1f}%")
print(f"  Ever contested general: {trad['ever_contested_general'].mean()*100:.1f}%")
print(f"  Ever contested any: {trad['ever_contested_any'].mean()*100:.1f}%")
print()

# Statistical test
if len(prog) > 0 and len(trad) > 0:
    # Chi-square test for contested general
    contingency = pd.crosstab(
        notable_with_elections[notable_with_elections['progressive'] | notable_with_elections['traditional']]['progressive'],
        notable_with_elections[notable_with_elections['progressive'] | notable_with_elections['traditional']]['ever_contested_general']
    )
    chi2, p_val, dof, expected = stats.chi2_contingency(contingency)
    print(f"Chi-square test (contested general): χ²={chi2:.3f}, p={p_val:.4f}")
    print()

# =============================================================================
# RESEARCH QUESTION 2: Recall Context Analysis  
# =============================================================================
print("="*80)
print("RQ2: WERE RECALLED PROSECUTORS MORE/LESS LIKELY TO WIN CLOSE ELECTIONS?")
print("="*80)
print()

# Recalled prosecutors
recalled = ['Chesa Boudin', 'Pamela Price']
recall_attempts = ['George Gascon']

notable_with_elections['recalled'] = notable_with_elections['name'].isin(recalled)
notable_with_elections['recall_attempts'] = notable_with_elections['name'].isin(recall_attempts)

print("RECALLED PROSECUTORS:")
for name in recalled:
    prosecutor = notable_with_elections[notable_with_elections['name'] == name]
    if len(prosecutor) > 0:
        p = prosecutor.iloc[0]
        print(f"\n{name}:")
        print(f"  Closest primary margin: {p['closest_primary_margin']:.1f}%" if pd.notna(p['closest_primary_margin']) else "  No primary data")
        print(f"  Closest general margin: {p['closest_general_margin']:.1f}%" if pd.notna(p['closest_general_margin']) else "  No general data")
        print(f"  Had close primary: {p['had_close_primary']}")
        print(f"  Had close general: {p['had_close_general']}")

print("\n\nPROSECUTORS WITH RECALL ATTEMPTS:")
for name in recall_attempts:
    prosecutor = notable_with_elections[notable_with_elections['name'] == name]
    if len(prosecutor) > 0:
        p = prosecutor.iloc[0]
        print(f"\n{name}:")
        print(f"  Closest primary margin: {p['closest_primary_margin']:.1f}%" if pd.notna(p['closest_primary_margin']) else "  No primary data")
        print(f"  Closest general margin: {p['closest_general_margin']:.1f}%" if pd.notna(p['closest_general_margin']) else "  No general data")
        print(f"  Had close primary: {p['had_close_primary']}")
        print(f"  Had close general: {p['had_close_general']}")

# Compare to non-recalled
non_recalled = notable_with_elections[~notable_with_elections['recalled'] & ~notable_with_elections['recall_attempts']]
print("\n\nNON-RECALLED PROSECUTORS (n={})".format(len(non_recalled)))
print(f"  Mean primary margin: {non_recalled['closest_primary_margin'].mean():.1f}%")
print(f"  Mean general margin: {non_recalled['closest_general_margin'].mean():.1f}%")
print(f"  % with close primary: {non_recalled['had_close_primary'].mean()*100:.1f}%")
print(f"  % with close general: {non_recalled['had_close_general'].mean()*100:.1f}%")
print()

# =============================================================================
# RESEARCH QUESTION 3: Challenger vs. Incumbent Familiarity
# =============================================================================
print("="*80)
print("RQ3: DOES FAMILIARITY DIFFER FOR CHALLENGERS VS. INCUMBENTS?")
print("="*80)
print()

# Notable prosecutors
challengers = notable_with_elections[notable_with_elections['ever_ran_as_challenger']]
incumbents = notable_with_elections[notable_with_elections['ever_ran_as_incumbent']]

print("NOTABLE PROSECUTORS:")
print(f"\nEver ran as challenger (n={len(challengers)}):")
print(f"  Mean familiarity: {challengers['familiarity_rate'].mean():.2f}%")
print(f"  Median familiarity: {challengers['familiarity_rate'].median():.2f}%")

print(f"\nEver ran as incumbent (n={len(incumbents)}):")
print(f"  Mean familiarity: {incumbents['familiarity_rate'].mean():.2f}%")
print(f"  Median familiarity: {incumbents['familiarity_rate'].median():.2f}%")

if len(challengers) > 0 and len(incumbents) > 0:
    t_stat, p_val = stats.ttest_ind(challengers['familiarity_rate'], 
                                     incumbents['familiarity_rate'])
    print(f"\nT-test: t={t_stat:.3f}, p={p_val:.4f}")
print()

# State prosecutors
state_with_elections = state_df[state_df['num_elections'] > 0].copy()
state_challengers = state_with_elections[state_with_elections['ever_ran_as_challenger']]
state_incumbents = state_with_elections[state_with_elections['ever_ran_as_incumbent']]

print("STATE PROSECUTORS:")
print(f"\nEver ran as challenger (n={len(state_challengers)}):")
print(f"  Mean familiarity: {state_challengers['familiarity_rate'].mean():.2f}%")
print(f"  Median familiarity: {state_challengers['familiarity_rate'].median():.2f}%")

print(f"\nEver ran as incumbent (n={len(state_incumbents)}):")
print(f"  Mean familiarity: {state_incumbents['familiarity_rate'].mean():.2f}%")
print(f"  Median familiarity: {state_incumbents['familiarity_rate'].median():.2f}%")

if len(state_challengers) > 0 and len(state_incumbents) > 0:
    t_stat, p_val = stats.ttest_ind(state_challengers['familiarity_rate'], 
                                     state_incumbents['familiarity_rate'])
    print(f"\nT-test: t={t_stat:.3f}, p={p_val:.4f}")
print()

# =============================================================================
# RESEARCH QUESTION 4: Multiple Elections and Familiarity
# =============================================================================
print("="*80)
print("RQ4: DOES FAMILIARITY INCREASE WITH NUMBER OF ELECTIONS?")
print("="*80)
print()

print("NOTABLE PROSECUTORS:")
corr, p_val = pearsonr(notable_with_elections['num_elections'], 
                        notable_with_elections['familiarity_rate'])
print(f"  Correlation: r={corr:.3f}, p={p_val:.4f}")
print(f"  Interpretation: {'Positive' if corr > 0 else 'Negative'} - " + 
      f"{'more elections = higher familiarity' if corr > 0 else 'more elections = lower familiarity'}")

# By tercile (use 3 groups instead of 4 due to small sample)
try:
    notable_with_elections['election_tercile'] = pd.qcut(notable_with_elections['num_elections'], 
                                                          q=3, labels=['Low', 'Med', 'High'],
                                                          duplicates='drop')
    print("\nFamiliarity by number of elections:")
    for q in ['Low', 'Med', 'High']:
        subset = notable_with_elections[notable_with_elections['election_tercile'] == q]
        if len(subset) > 0:
            print(f"  {q} (n={len(subset)}): Mean familiarity = {subset['familiarity_rate'].mean():.2f}%")
except:
    print("\nCannot stratify - too few unique values")

print("\n\nSTATE PROSECUTORS:")
corr, p_val = pearsonr(state_with_elections['num_elections'], 
                        state_with_elections['familiarity_rate'])
print(f"  Correlation: r={corr:.3f}, p={p_val:.4f}")
print(f"  Interpretation: {'Positive' if corr > 0 else 'Negative'} - " + 
      f"{'more elections = higher familiarity' if corr > 0 else 'more elections = lower familiarity'}")

# By quartile
try:
    state_with_elections['election_quartile'] = pd.qcut(state_with_elections['num_elections'], 
                                                         q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'],
                                                         duplicates='drop')
    print("\nFamiliarity by number of elections:")
    for q in ['Q1', 'Q2', 'Q3', 'Q4']:
        subset = state_with_elections[state_with_elections['election_quartile'] == q]
        if len(subset) > 0:
            print(f"  {q} (n={len(subset)}): Mean familiarity = {subset['familiarity_rate'].mean():.2f}%")
except:
    print("\nCannot stratify - too few unique values")
print()

# =============================================================================
# RESEARCH QUESTION 5: Primary vs. General Effects by Ideology
# =============================================================================
print("="*80)
print("RQ5: DO PROGRESSIVE VS TRADITIONAL PROSECUTORS GAIN VISIBILITY DIFFERENTLY?")
print("="*80)
print()

print("HYPOTHESIS: Progressive prosecutors gain visibility from contested primaries")
print("           Traditional prosecutors gain visibility from contested generals")
print()

# Notable prosecutors - progressive group
prog_contested_prim = prog[prog['ever_contested_primary']]
prog_uncontested_prim = prog[~prog['ever_contested_primary']]

print("PROGRESSIVE PROSECUTORS:")
print(f"  Contested primary (n={len(prog_contested_prim)}): {prog_contested_prim['familiarity_rate'].mean():.2f}% familiarity")
print(f"  Uncontested primary (n={len(prog_uncontested_prim)}): {prog_uncontested_prim['familiarity_rate'].mean():.2f}% familiarity")
if len(prog_contested_prim) > 0 and len(prog_uncontested_prim) > 0:
    diff = prog_contested_prim['familiarity_rate'].mean() - prog_uncontested_prim['familiarity_rate'].mean()
    print(f"  Difference: {diff:+.2f}%")

prog_contested_gen = prog[prog['ever_contested_general']]
prog_uncontested_gen = prog[~prog['ever_contested_general']]

print(f"\n  Contested general (n={len(prog_contested_gen)}): {prog_contested_gen['familiarity_rate'].mean():.2f}% familiarity")
print(f"  Uncontested general (n={len(prog_uncontested_gen)}): {prog_uncontested_gen['familiarity_rate'].mean():.2f}% familiarity")
if len(prog_contested_gen) > 0 and len(prog_uncontested_gen) > 0:
    diff = prog_contested_gen['familiarity_rate'].mean() - prog_uncontested_gen['familiarity_rate'].mean()
    print(f"  Difference: {diff:+.2f}%")

# Traditional prosecutors
trad_contested_prim = trad[trad['ever_contested_primary']]
trad_uncontested_prim = trad[~trad['ever_contested_primary']]

print("\n\nTRADITIONAL PROSECUTORS:")
print(f"  Contested primary (n={len(trad_contested_prim)}): {trad_contested_prim['familiarity_rate'].mean():.2f}% familiarity")
print(f"  Uncontested primary (n={len(trad_uncontested_prim)}): {trad_uncontested_prim['familiarity_rate'].mean():.2f}% familiarity")
if len(trad_contested_prim) > 0 and len(trad_uncontested_prim) > 0:
    diff = trad_contested_prim['familiarity_rate'].mean() - trad_uncontested_prim['familiarity_rate'].mean()
    print(f"  Difference: {diff:+.2f}%")

trad_contested_gen = trad[trad['ever_contested_general']]
trad_uncontested_gen = trad[~trad['ever_contested_general']]

print(f"\n  Contested general (n={len(trad_contested_gen)}): {trad_contested_gen['familiarity_rate'].mean():.2f}% familiarity")
print(f"  Uncontested general (n={len(trad_uncontested_gen)}): {trad_uncontested_gen['familiarity_rate'].mean():.2f}% familiarity")
if len(trad_contested_gen) > 0 and len(trad_uncontested_gen) > 0:
    diff = trad_contested_gen['familiarity_rate'].mean() - trad_uncontested_gen['familiarity_rate'].mean()
    print(f"  Difference: {diff:+.2f}%")

print()

# =============================================================================
# SUMMARY VISUALIZATION
# =============================================================================

# Create visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# RQ3: Challenger vs Incumbent - Notable
ax = axes[0, 0]
data = [challengers['familiarity_rate'].dropna(), incumbents['familiarity_rate'].dropna()]
ax.boxplot(data, labels=['Challenger', 'Incumbent'])
ax.set_ylabel('Familiarity Rate (%)')
ax.set_title('RQ3: Notable Prosecutors\nChallenger vs. Incumbent', fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# RQ3: Challenger vs Incumbent - State
ax = axes[0, 1]
data = [state_challengers['familiarity_rate'].dropna(), state_incumbents['familiarity_rate'].dropna()]
ax.boxplot(data, labels=['Challenger', 'Incumbent'])
ax.set_ylabel('Familiarity Rate (%)')
ax.set_title('RQ3: State Prosecutors\nChallenger vs. Incumbent', fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# RQ4: Number of Elections - Notable
ax = axes[0, 2]
ax.scatter(notable_with_elections['num_elections'], 
          notable_with_elections['familiarity_rate'], 
          alpha=0.6, s=80)
z = np.polyfit(notable_with_elections['num_elections'], 
               notable_with_elections['familiarity_rate'], 1)
p = np.poly1d(z)
x_line = np.linspace(notable_with_elections['num_elections'].min(), 
                     notable_with_elections['num_elections'].max(), 100)
ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)
ax.set_xlabel('Number of Elections')
ax.set_ylabel('Familiarity Rate (%)')
ax.set_title('RQ4: Notable Prosecutors\nElections vs. Familiarity', fontweight='bold')
ax.grid(alpha=0.3)

# RQ4: Number of Elections - State
ax = axes[1, 0]
ax.scatter(state_with_elections['num_elections'], 
          state_with_elections['familiarity_rate'], 
          alpha=0.3, s=40)
z = np.polyfit(state_with_elections['num_elections'], 
               state_with_elections['familiarity_rate'], 1)
p = np.poly1d(z)
x_line = np.linspace(state_with_elections['num_elections'].min(), 
                     state_with_elections['num_elections'].max(), 100)
ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)
ax.set_xlabel('Number of Elections')
ax.set_ylabel('Familiarity Rate (%)')
ax.set_title('RQ4: State Prosecutors\nElections vs. Familiarity', fontweight='bold')
ax.grid(alpha=0.3)

# RQ5: Progressive Primary vs General
ax = axes[1, 1]
prog_data = [
    prog_contested_prim['familiarity_rate'].dropna(),
    prog_uncontested_prim['familiarity_rate'].dropna(),
    prog_contested_gen['familiarity_rate'].dropna(),
    prog_uncontested_gen['familiarity_rate'].dropna()
]
positions = [1, 2, 4, 5]
bp = ax.boxplot(prog_data, positions=positions)
ax.set_xticks(positions)
ax.set_xticklabels(['Cont.\nPrim', 'Uncont.\nPrim', 'Cont.\nGen', 'Uncont.\nGen'], fontsize=9)
ax.set_ylabel('Familiarity Rate (%)')
ax.set_title('RQ5: Progressive Prosecutors\nPrimary vs. General Effects', fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# RQ5: Traditional Primary vs General
ax = axes[1, 2]
trad_data = [
    trad_contested_prim['familiarity_rate'].dropna(),
    trad_uncontested_prim['familiarity_rate'].dropna(),
    trad_contested_gen['familiarity_rate'].dropna(),
    trad_uncontested_gen['familiarity_rate'].dropna()
]
positions = [1, 2, 4, 5]
bp = ax.boxplot(trad_data, positions=positions)
ax.set_xticks(positions)
ax.set_xticklabels(['Cont.\nPrim', 'Uncont.\nPrim', 'Cont.\nGen', 'Uncont.\nGen'], fontsize=9)
ax.set_ylabel('Familiarity Rate (%)')
ax.set_title('RQ5: Traditional Prosecutors\nPrimary vs. General Effects', fontweight='bold')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('output/tier1_research_questions.png', dpi=300, bbox_inches='tight')
print("✓ Visualization saved: tier1_research_questions.png")
print()

print("="*80)
print("TIER 1 ANALYSIS COMPLETE")
print("="*80)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PROSECUTOR IDEOLOGY & FAMILIARITY — MASTER FINAL
================================================
This script unifies the corrected pipeline and the most useful analyses/visualizations
from prior versions into ONE reproducible entry-point.

Key CORRECT rules implemented:
- SCALE is 1–4: {Very Traditional:1, Traditional:2, Progressive:3, Very Progressive:4}
- 'Not Familiar' is missing (NaN) for ideology computations
- NATIONAL (50 Notables) familiarity denominator = respondents who COMPLETED the
  national section (the 'more' gate) — expected 407 in current data.
- STATE familiarity denominator = number of in-state respondents who actually
  engaged with state DA items (>=1 state DA response).
- Modeling subset requires >=10 substantive ratings for ideology stability.

Outputs:
- CSVs in outputs/corrected/
- Figures in outputs/corrected/visualizations/
- A plain-text summary with key counts and file pointers.
"""
import os, sys, re, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = True

warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# 0) Robust Paths & Config
# -----------------------------------------------------------------------------

class Config:
    SURVEY_FILE = 'PP+survey+draft_October+16,+2025_13.32.csv'
    ELECTION_FILE = 'elections_with_reconciled_contested.csv'
    OUTPUT_DIR = Path('outputs/corrected')
    VIZ_DIR = OUTPUT_DIR / 'visualizations'

    MIN_RATINGS_THRESHOLD = 10
    CLOSE_MARGIN_THRESHOLD = 55  # % (<= this is "close")

    RATING_MAP = {
        'Very Traditional': 1,
        'Traditional': 2,
        'Not Familiar': np.nan,
        'Progressive': 3,
        'Very Progressive': 4
    }

# -----------------------------------------------------------------------------
# 1) Load the proven corrected pipeline classes from your latest script
#    (we execute it into a namespace to import its classes without renaming file)
# -----------------------------------------------------------------------------

LATEST_SCRIPT = 'FINAL_prosecutor_analysis_master_CORRECTED (1).py'

ns = {}
with open(LATEST_SCRIPT, 'r', encoding='utf-8') as f:
    src = f.read()
exec(src, ns)

DataLoader = ns['DataLoader']
ProsecutorAnalyzer = ns['ProsecutorAnalyzer']

# -----------------------------------------------------------------------------
# 2) Orchestrator: run the canonical pipeline + add visualizations & extras
# -----------------------------------------------------------------------------

def run_master():
    # Ensure output dirs
    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Config.VIZ_DIR.mkdir(parents=True, exist_ok=True)

    # Load & analyze using the canonical (corrected) logic
    loader = DataLoader(Config.SURVEY_FILE, Config.ELECTION_FILE).load()
    an = ProsecutorAnalyzer(loader)

    # Canonical steps from corrected script
    an.build_prosecutor_dataset()
    an.match_with_elections()
    an.analyze_familiarity_patterns()
    an.analyze_incumbency()
    an.analyze_contestation()
    an.analyze_recall_risk()

    # Multivariate models (guarded)
    try:
        an.run_multivariate_models()
    except Exception as e:
        print("[Note] run_multivariate_models raised an exception and was skipped:", repr(e))

    # Export baseline corrected CSVs & summary (provided by corrected script)
    an.export_results(Config.OUTPUT_DIR)

    # ------------------------------------------------------------------
    # 2A) Add a clean visualization suite (matplotlib-only)
    # ------------------------------------------------------------------
    # We will use the dataframes the corrected analyzer exposes:
    # - an.df_prosecutors               (all)
    # - an.df_prosecutors_filtered_ideology
    # - an.df_matched_all               (matched for familiarity/elections)
    # - an.df_matched                   (filtered >=10 for ideology models)

    # Helper: safe save
    def savefig(path):
        plt.tight_layout()
        plt.savefig(path, dpi=200, bbox_inches='tight')
        plt.close()

    # FIG 1: Familiarity (means) – Notables vs State (bar)
    try:
        df = an.df_prosecutors
        notable = df[df['is_notable']]
        state = df[~df['is_notable']]
        mean_notable = notable['familiarity_rate'].mean()
        mean_state = state['familiarity_rate'].mean()

        fig1_path = Config.VIZ_DIR / 'fig1_familiarity_notables_vs_state.png'
        fig = plt.figure(figsize=(5,4))
        xs = ['Notables', 'State']
        ys = [mean_notable, mean_state]
        plt.bar(xs, ys)
        plt.ylabel('Mean familiarity (%)')
        plt.title('Familiarity: Notables vs State')
        savefig(fig1_path)
    except Exception as e:
        print("[Viz] FIG 1 failed:", repr(e))

    # FIG 2: Top/Bottom 10 Notables by Mean Ideology (barh)
    try:
        notables = an.df_prosecutors[an.df_prosecutors['is_notable']].copy()
        notables = notables[pd.notna(notables['mean_score'])]
        top10 = notables.sort_values('mean_score', ascending=False).head(10)
        bot10 = notables.sort_values('mean_score', ascending=True).head(10)

        def plot_barh(dfsub, title, path):
            plt.figure(figsize=(7,5))
            names = dfsub['name']
            vals = dfsub['mean_score']
            y = np.arange(len(dfsub))
            plt.barh(y, vals)
            plt.yticks(y, names)
            plt.xlabel('Mean ideology (1=Trad → 4=Prog)')
            plt.title(title)
            savefig(path)

        plot_barh(top10.iloc[::-1], 'Top 10 Notables (Most Progressive)', Config.VIZ_DIR / 'fig2a_top10_notables.png')
        plot_barh(bot10.iloc[::-1], 'Bottom 10 Notables (Most Traditional)', Config.VIZ_DIR / 'fig2b_bottom10_notables.png')
    except Exception as e:
        print("[Viz] FIG 2 failed:", repr(e))

    # FIG 3: Familiarity vs Closest General Margin (scatter with OLS line)
    try:
        dfm = an.df_matched_all.copy()
        dfm = dfm[pd.notna(dfm['closest_general_margin']) & pd.notna(dfm['familiarity_rate'])]
        if len(dfm) >= 3:
            x = dfm['closest_general_margin'].values
            y = dfm['familiarity_rate'].values
            A = np.vstack([x, np.ones_like(x)]).T
            coef, resid, rank, s = np.linalg.lstsq(A, y, rcond=None)
            xs = np.linspace(x.min(), x.max(), 100)
            ys = coef[0]*xs + coef[1]

            plt.figure(figsize=(6,4))
            plt.scatter(x, y, alpha=0.7)
            plt.plot(xs, ys)
            plt.xlabel('Closest general margin (%)')
            plt.ylabel('Familiarity (%)')
            plt.title('Familiarity vs Electoral Competitiveness')
            savefig(Config.VIZ_DIR / 'fig3_familiarity_vs_margin.png')
    except Exception as e:
        print("[Viz] FIG 3 failed:", repr(e))

    # FIG 4: Ideology summary (mean ± sd) for top 6 Notables by familiarity
    try:
        df = an.df_prosecutors[an.df_prosecutors['is_notable']].copy()
        top6 = df.sort_values('familiarity_rate', ascending=False).head(6)
        plt.figure(figsize=(7,4))
        x = np.arange(len(top6))
        means = top6['mean_score'].fillna(0).values
        sds = top6['sd_score'].fillna(0).values if 'sd_score' in top6 else np.zeros_like(means)
        plt.bar(x, means, yerr=sds, capsize=4)
        plt.xticks(x, top6['name'], rotation=30, ha='right')
        plt.ylabel('Mean ideology (1=Trad → 4=Prog)')
        plt.title('Ideology (mean ± sd): Top-6 Familiar Notables')
        savefig(Config.VIZ_DIR / 'fig4_ideology_summary_top6.png')
    except Exception as e:
        print("[Viz] FIG 4 failed:", repr(e))

    # FIG 5: Contestation rates by ideology bucket
    try:
        dfm = an.df_matched_all.copy()
        dfm['bucket'] = np.where(
            dfm['mean_score']>=3.0, 'Progressive-ish (≥3.0)',
            np.where(dfm['mean_score']<=2.0, 'Traditional-ish (≤2.0)', 'Middle')
        )
        rates = dfm.groupby('bucket')['ever_contested_any'].mean().reindex(
            ['Traditional-ish (≤2.0)','Middle','Progressive-ish (≥3.0)']
        )
        plt.figure(figsize=(6,4))
        plt.bar(rates.index, rates.values*100.0)
        plt.ylabel('Ever contested (any) — %')
        plt.title('Contestation vs Ideology bucket')
        plt.xticks(rotation=20)
        savefig(Config.VIZ_DIR / 'fig5_contestation_vs_ideology.png')
    except Exception as e:
        print("[Viz] FIG 5 failed:", repr(e))

    # FIG 6: Recalled prosecutors — margin context bars
    try:
        dfm = an.df_matched_all.copy()
        recalled = dfm[dfm['recalled']==True].copy()
        if len(recalled)>0:
            plt.figure(figsize=(6,3.5))
            names = recalled['name']
            vals = recalled['closest_general_margin'].fillna(0)
            y = np.arange(len(recalled))
            plt.barh(y, vals)
            plt.yticks(y, names)
            plt.xlabel('Closest general election margin (%)')
            plt.title('Recall context: margins')
            savefig(Config.VIZ_DIR / 'fig6_recalled_margins.png')
    except Exception as e:
        print("[Viz] FIG 6 failed:", repr(e))

    # ------------------------------------------------------------------
    # 2B) Write an augmented summary txt
    # ------------------------------------------------------------------
    out = []
    out.append("="*78)
    out.append("MASTER FINAL RUN — SUMMARY")
    out.append("="*78)
    out.append(f"Respondents (after consent): {loader.total_respondents}")
    out.append(f"Completed national section (expected ~407): {loader.completed_national_section}")
    out.append("")
    out.append(f"All prosecutors: {len(an.df_prosecutors)}  | Notables: {int(an.df_prosecutors['is_notable'].sum())}  | State: {int((~an.df_prosecutors['is_notable']).sum())}")
    out.append(f"Mean familiarity — Notables: {an.df_prosecutors[an.df_prosecutors['is_notable']]['familiarity_rate'].mean():.2f}%")
    out.append(f"Mean familiarity — State: {an.df_prosecutors[~an.df_prosecutors['is_notable']]['familiarity_rate'].mean():.2f}%")
    out.append("")
    out.append(f"Matched (FULL): {len(an.df_matched_all)} | Notable: {int(an.df_matched_all['is_notable'].sum())} | State: {int((~an.df_matched_all['is_notable']).sum())}")
    out.append(f"Matched (FILTERED ≥{Config.MIN_RATINGS_THRESHOLD}): {len(an.df_matched)}")
    out.append("")
    out.append("Exports:")
    out.append("  - outputs/corrected/all_prosecutors_CORRECTED.csv")
    out.append("  - outputs/corrected/prosecutors_filtered_ideology_CORRECTED.csv")
    out.append("  - outputs/corrected/matched_prosecutors_FULL_CORRECTED.csv")
    out.append("  - outputs/corrected/matched_prosecutors_FILTERED_CORRECTED.csv")
    out.append("  - outputs/corrected/CORRECTED_FINDINGS_SUMMARY.txt")
    out.append("  - outputs/corrected/visualizations/*.png")
    outp = Config.OUTPUT_DIR / "MASTER_FINAL_SUMMARY.txt"
    outp.write_text("\n".join(out), encoding='utf-8')

if __name__ == "__main__":
    run_master()

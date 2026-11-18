import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import evaluate, dimensions

DATA_PATH = 'Data/Dataset-CityIndicators-90%.csv'
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'oat_outputs')
PROFILE = (1, 2, 3, 4)
LEVELS = [0.05, 0.10]
N_REPS = 30
SEED = 42
TOP_K = 10

NEGATIVE = {
    'Gini index of income inequality', 'housing burden', 'rent burden',
    '% ppl below poverty lvl', 'student/teacher ratio',
    'Mortality (deaths per 100,000)', 'crime rate per 100,000 ppl',
    'mean commute time to work (min)',
}

BOUNDED_0_100 = {
    'employment / population ratio (16 yrs & older)',
    '% ppl below poverty lvl',
    '% ppl 18-24: less than high school graduate',
    '% ppl 18-24: high school graduate',
    "% ppl 18-24: some college/associate's degree",
    "% ppl 18-24: bachelor's degree or higher",
    '% pop enrolled in college or grad school',
    '% ppl insured', '% routine checkup', 'walkability index',
}
MODEL_INDICATORS = [ind for inds in dimensions.values() for ind in inds]

def load_data(path=DATA_PATH):
    df = pd.read_csv(path, header=1, skiprows=[2])
    df = df.dropna(subset=['Geographic Area Name']).reset_index(drop=True)
    return df

def ranks_of(results):
    return pd.Series(np.arange(1, len(results) + 1), index=results['City'].values)

def clip_bounded(df, col):
    if col in BOUNDED_0_100:
        df[col] = df[col].clip(0, 100)

def compare(base_ranks, new_ranks, base_top, base_first):
    new_ranks = new_ranks.reindex(base_ranks.index)
    diff = (new_ranks - base_ranks).abs()
    return {
        'spearman': base_ranks.corr(new_ranks, method='spearman'),
        'top10_overlap': len(base_top & set(new_ranks.nsmallest(TOP_K).index)),
        'max_shift': diff.max(),
        'mean_shift': diff.mean(),
        'top1_same': float(new_ranks.idxmin() == base_first),
    }


# ----------------------------- 1. indicator OAT --------------------------
def indicator_oat(df, base_ranks, base_top, base_first, rng):
    rows = []
    for ind in MODEL_INDICATORS:
        raw = pd.to_numeric(df[ind], errors='coerce')
        for delta in LEVELS:
            runs = []
            for _ in range(N_REPS):
                noise = rng.uniform(-delta, delta, size=len(df))
                d = df.copy()
                d[ind] = raw * (1 + noise)
                clip_bounded(d, ind)
                runs.append(compare(base_ranks, ranks_of(evaluate(d, PROFILE)),
                                    base_top, base_first))
            r = pd.DataFrame(runs)
            rows.append({
                'indicator': ind,
                'delta': delta,
                'spearman_mean': r['spearman'].mean(),
                'spearman_min': r['spearman'].min(),
                'top10_overlap_mean': r['top10_overlap'].mean(),
                'top10_overlap_min': r['top10_overlap'].min(),
                'max_shift_mean': r['max_shift'].mean(),
                'max_shift_worst': r['max_shift'].max(),
                'mean_shift_mean': r['mean_shift'].mean(),
                'top1_same_share': r['top1_same'].mean(),
            })
    return pd.DataFrame(rows)


# ------------------------------- 2. city OAT -----------------------------
def city_oat(df, base_ranks, base_top, base_first):
    rows = []
    for i, city in enumerate(df['Geographic Area Name']):
        for delta in LEVELS:
            for direction, sign in (('improve', 1), ('worsen', -1)):
                d = df.copy()
                for ind in MODEL_INDICATORS:
                    d[ind] = pd.to_numeric(d[ind], errors='coerce')
                    # "better" = higher for positive indicators, lower for negative ones
                    factor = 1 + sign * delta * (-1 if ind in NEGATIVE else 1)
                    d.loc[i, ind] = d.loc[i, ind] * factor
                    clip_bounded(d, ind)
                new_ranks = ranks_of(evaluate(d, PROFILE)).reindex(base_ranks.index)
                others = new_ranks.drop(city) - base_ranks.drop(city)
                m = compare(base_ranks, new_ranks, base_top, base_first)
                rows.append({
                    'city': city,
                    'direction': direction,
                    'delta': delta,
                    'base_rank': base_ranks[city],
                    'new_rank': new_ranks[city],
                    'rank_change': base_ranks[city] - new_ranks[city],   # + = moved up
                    'others_mean_shift': others.abs().mean(),
                    'others_max_shift': others.abs().max(),
                    'spearman': m['spearman'],
                    'top10_overlap': m['top10_overlap'],
                    'top1_same': m['top1_same'],
                })
    return pd.DataFrame(rows)


# --------------------------------- plots ---------------------------------
INK, MUTED = '#1f2933', '#6b7785'
C_LIGHT, C_DARK = '#9ec5ee', '#1f6fbf'


def style_axes(ax):
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.spines['bottom'].set_color('#c9d1d9')
    ax.tick_params(colors=MUTED, length=0)
    ax.xaxis.grid(True, color='#e6eaee', linewidth=0.8)
    ax.set_axisbelow(True)


def plot_indicator(ind_df, path):
    piv = ind_df.pivot(index='indicator', columns='delta', values='mean_shift_mean')
    piv = piv.sort_values(LEVELS[-1])
    fig, ax = plt.subplots(figsize=(9, 7))
    y = np.arange(len(piv))
    h = 0.38
    b1 = ax.barh(y + h / 2, piv[LEVELS[0]], h, color=C_LIGHT, label=f'±{LEVELS[0]:.0%} noise')
    b2 = ax.barh(y - h / 2, piv[LEVELS[-1]], h, color=C_DARK, label=f'±{LEVELS[-1]:.0%} noise')
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_width() + 0.02, b.get_y() + b.get_height() / 2,
                    f'{b.get_width():.2f}', va='center', fontsize=8, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(piv.index, fontsize=9, color=INK)
    ax.set_xlabel('Mean absolute rank shift per city (avg. over runs)', color=MUTED)
    ax.set_title('Indicator OAT: which single indicator moves the ranking most', loc='left',
                 color=INK, fontsize=12, fontweight='bold')
    ax.legend(frameon=False, loc='lower right')
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_city(city_df, path, delta=LEVELS[-1], n=15):
    s = city_df[(city_df.delta == delta) & (city_df.direction == 'improve')]
    s = s.reindex(s['rank_change'].abs().sort_values(ascending=False).index).head(n)
    s = s.sort_values('rank_change')
    fig, ax = plt.subplots(figsize=(9, 6))
    short = (s['city'].str.replace(r' \(balance\)', '', regex=True)
                      .str.replace(r' (city|town|CDP|municipality)(?=,)', '', regex=True)
                      .str.replace(r'/Jefferson County metro government', '', regex=True))
    ax.barh(short, s['rank_change'], color=C_DARK, height=0.6)
    for yy, (v, b) in enumerate(zip(s['rank_change'], s['base_rank'])):
        ax.text(v + 0.2, yy, f'{int(v)} (from #{int(b)})', va='center', fontsize=8, color=INK)
    ax.set_xlabel('Rank places gained', color=MUTED)
    style_axes(ax)
    ax.tick_params(axis='y', labelsize=9, labelcolor=INK)
    ax.set_xlim(0, s['rank_change'].max() * 1.25)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.suptitle(f'City OAT: biggest rank gains when one city improves {delta:.0%} on all indicators',
                 x=0.02, ha='left', color=INK, fontsize=12, fontweight='bold')
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ---------------------------------- main ---------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)
    df = load_data()

    base = evaluate(df, PROFILE)
    base_ranks = ranks_of(base)
    base_top = set(base_ranks.nsmallest(TOP_K).index)
    base_first = base_ranks.idxmin()
    base.to_csv(os.path.join(OUT_DIR, 'baseline_ranking.csv'), index=False)

    ind_df = indicator_oat(df, base_ranks, base_top, base_first, rng)
    ind_df = ind_df.sort_values(['delta', 'mean_shift_mean'], ascending=[True, False])
    ind_df.to_csv(os.path.join(OUT_DIR, 'indicator_oat_summary.csv'), index=False)

    city_df = city_oat(df, base_ranks, base_top, base_first)
    city_df.to_csv(os.path.join(OUT_DIR, 'city_oat_summary.csv'), index=False)

    plot_indicator(ind_df, os.path.join(OUT_DIR, 'indicator_oat.png'))
    plot_city(city_df, os.path.join(OUT_DIR, 'city_oat.png'))

    pd.set_option('display.width', 200)
    pd.set_option('display.max_columns', None)
    print(f'Profile {PROFILE}; baseline #1: {base_first}\n')
    for delta in LEVELS:
        print(f'--- Indicator OAT, ±{delta:.0%} noise (most influential first) ---')
        print(ind_df[ind_df.delta == delta].drop(columns='delta').round(3).to_string(index=False))
        print()
    for direction in ('improve', 'worsen'):
        s = city_df[(city_df.direction == direction)]
        print(f'--- City OAT, {direction}: summary of own-rank change by delta ---')
        print(s.groupby('delta')['rank_change'].agg(['mean', 'min', 'max']).round(2))
        print('top-10 membership changes:',
              s.groupby('delta')['top10_overlap'].apply(lambda x: (x < TOP_K).sum()).to_dict(), '\n')


if __name__ == '__main__':
    main()

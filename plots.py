import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# Import from main.py
from main import (
    sample_C_set, get_seeds, MASTER_SEED,
    make_initial_X, make_W_symmetric, make_W_stratified, make_G,
    INIT_MEANS, TOPIC_NAMES, CLASS_NAMES,
    N_PER_CLASS, N_TOTAL, N_REPS, LAM, EXPERIMENTS,
)


def make_all_figures(results):
    """Generate all result figures."""
    _fig1_trajectories(results)
    _fig2_suppression(results)
    _fig3_interaction(results)
    _fig4_shift(results)
    print("  All figures saved.")



def _fig1_trajectories(results):
    seeds = get_seeds(n_reps=1, master_seed=MASTER_SEED)
    COLORS = ['#d62728', '#2ca02c', '#1f77b4']

    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True, sharey=True)
    for row, (wt, label, ek) in enumerate([
        ('symmetric', 'E1: Symmetric, no suppression', 'E1'),
        ('stratified', 'E5: Stratified, no suppression', 'E5'),
    ]):
        rng_c = np.random.default_rng(seeds['C'][0])
        C_set = sample_C_set(rng_c)
        C_all = np.zeros((N_TOTAL, 3, 3)); idx = 0
        for cls, n in enumerate(N_PER_CLASS):
            for _ in range(n): C_all[idx] = C_set[cls]; idx += 1
        X0 = make_initial_X(N_PER_CLASS, seed=int(seeds['X'][0]))
        W = make_W_stratified(N_PER_CLASS, seed=int(seeds['W'][0])) if wt == 'stratified' \
            else make_W_symmetric(N_TOTAL, seed=int(seeds['W'][0]))
        G = make_G(); GtG = G.T @ G; I_GtG = np.eye(3) - GtG
        starts = [0]
        for n in N_PER_CLASS: starts.append(starts[-1] + n)
        X = X0.copy(); U = X0.copy(); traj = []
        for t in range(80):
            traj.append(np.array([X[starts[c]:starts[c+1]].mean(axis=0) for c in range(3)]))
            X = LAM * np.einsum('ijk,ik->ij', C_all, X @ I_GtG + W @ X @ GtG) + (1-LAM) * U
        traj.append(np.array([X[starts[c]:starts[c+1]].mean(axis=0) for c in range(3)]))
        traj = np.array(traj)
        for k in range(3):
            ax = axes[row, k]
            for c in range(3): ax.plot(traj[:, c, k], color=COLORS[c], lw=2, label=CLASS_NAMES[c])
            ax.axhline(0, color='gray', lw=0.5, ls='--'); ax.set_ylim(-1.05, 1.05)
            div_val = results[ek]['div_mean'][k]
            ax.set_title(f"{label}\n{TOPIC_NAMES[k]} (div={div_val:.3f})", fontsize=10)
            ax.grid(alpha=0.3)
            if k == 0: ax.set_ylabel('Class mean opinion')
            if row == 1: ax.set_xlabel('Time step')
            if row == 0 and k == 0: ax.legend(fontsize=8)
    plt.suptitle('Figure 1: Baseline trajectories — Symmetric vs Stratified',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout(); plt.savefig('fig1_trajectories.png', dpi=150, bbox_inches='tight')
    print("  Saved fig1_trajectories.png")


def _fig2_suppression(results):
    e1 = results['E1']['div_mean']
    cols = {'E1': '#4A90D9', 'E2': '#F5A623', 'E3': '#7ED321', 'E4': '#D0021B'}
    labs = {'E1': 'No suppr', 'E2': 'Suppr Opp', 'E3': 'Suppr Health', 'E4': 'Suppr Wealth'}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    x = np.arange(3); bw = 0.2
    ax = axes[0]
    for i, e in enumerate(['E1', 'E2', 'E3', 'E4']):
        ax.bar(x + (i - 1.5) * bw, results[e]['div_mean'], bw,
               yerr=results[e]['div_std'], capsize=3,
               color=cols[e], label=labs[e], alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(TOPIC_NAMES, fontsize=12)
    ax.set_ylabel('Inter-class Divergence', fontsize=12)
    ax.set_title('Per-topic divergence', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(alpha=0.3, axis='y'); ax.set_ylim(0, 0.7)

    ax = axes[1]
    for k in range(3):
        for i, e in enumerate(['E2', 'E3', 'E4']):
            d = results[e]['div_mean'][k] - e1[k]
            ax.bar(x[k] + (i - 1) * 0.22, d, 0.22, color=cols[e], alpha=0.85)
            if abs(d) > 0.015:
                ax.text(x[k] + (i - 1) * 0.22, d + 0.005, f'{d:+.02f}',
                        ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(TOPIC_NAMES, fontsize=12)
    ax.set_ylabel('Δ Divergence (vs baseline)', fontsize=12)
    ax.set_title('Extra divergence from suppression', fontsize=12, fontweight='bold')
    ax.axhline(0, color='gray', lw=0.8); ax.grid(alpha=0.3, axis='y')
    ax.legend(handles=[Patch(facecolor=cols[e], label=labs[e]) for e in ['E2', 'E3', 'E4']],
              fontsize=9, loc='upper left')
    plt.suptitle('Figure 2: Which topic to suppress matters (symmetric network)',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout(); plt.savefig('fig2_suppression.png', dpi=150, bbox_inches='tight')
    print("  Saved fig2_suppression.png")


def _fig3_interaction(results):
    e1, e5 = results['E1']['div_mean'], results['E5']['div_mean']
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    ax = axes[0]
    sv = [results[e]['div_mean'].mean() for e in ['E1', 'E2', 'E3', 'E4']]
    rv = [results[e]['div_mean'].mean() for e in ['E5', 'E6', 'E7', 'E8']]
    x = np.arange(4); w = 0.35
    ax.bar(x - w/2, sv, w, color='#4A90D9', label='Symmetric', alpha=0.85)
    ax.bar(x + w/2, rv, w, color='#D0021B', label='Stratified', alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(['None', 'Opp', 'Health', 'Wealth'], fontsize=12)
    ax.set_ylabel('Mean Divergence', fontsize=12)
    ax.set_title('All 8 experiments', fontsize=12, fontweight='bold')
    ax.legend(fontsize=11); ax.grid(alpha=0.3, axis='y')

    ax = axes[1]
    topics = ['Opportunity', 'Health', 'Wealth']
    gs = [(results[e]['div_mean'] - e1).mean() for e in ['E2', 'E3', 'E4']]
    gr = [(results[e]['div_mean'] - e5).mean() for e in ['E6', 'E7', 'E8']]
    x = np.arange(3)
    ax.bar(x - w/2, gs, w, color='#4A90D9', label='Symmetric', alpha=0.85)
    ax.bar(x + w/2, gr, w, color='#D0021B', label='Stratified', alpha=0.85)
    for i in range(3):
        ax.text(x[i] - w/2, gs[i] + 0.003, f'{gs[i]:.3f}', ha='center', fontsize=10)
        ax.text(x[i] + w/2, gr[i] + 0.003, f'{gr[i]:.3f}', ha='center', fontsize=10)
        ratio = gr[i] / gs[i] if abs(gs[i]) > 0.001 else 0
        c = '#D0021B' if ratio > 1 else '#4A90D9'
        ax.text(x[i], max(gs[i], gr[i]) + 0.015, f'{ratio:.1f}×',
                ha='center', fontsize=14, fontweight='bold', color=c)
    ax.axhline(0, color='gray', lw=0.8); ax.set_xticks(x); ax.set_xticklabels(topics, fontsize=12)
    ax.set_ylabel('Suppression effect', fontsize=12)
    ax.set_title('By network type', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(alpha=0.3, axis='y')
    plt.suptitle('Figure 3: Network × suppression interaction',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout(); plt.savefig('fig3_interaction.png', dpi=150, bbox_inches='tight')
    print("  Saved fig3_interaction.png")


def _fig4_shift(results):
    en_list = list(EXPERIMENTS.keys())
    ratios, labels, colors = [], [], []
    for en in en_list:
        s = results[en]['shift_mean'].sum(axis=1)
        r = s[2] / s[0] if s[0] > 0.01 else 99
        ratios.append(r)
        sp = EXPERIMENTS[en][1]
        labels.append(f"{en}\n{'No suppr' if sp is None else 'Suppr ' + TOPIC_NAMES[sp][:3]}")
        colors.append('#D0021B' if r > 2 else '#4A90D9' if r < 0.8 else '#888888')

    fig, ax = plt.subplots(figsize=(12, 5.5))
    x = np.arange(8)
    ax.bar(x, ratios, color=colors, alpha=0.85, edgecolor='white', linewidth=1.5)
    for i, r in enumerate(ratios):
        ax.text(i, r + 0.08, f'{r:.1f}×', ha='center', fontsize=12, fontweight='bold')
    ax.axhline(1.0, color='black', lw=1.5, ls='--', alpha=0.5)
    ax.axvline(3.5, color='gray', lw=1, ls=':', alpha=0.5)
    ax.text(1.5, max(ratios) * 0.95, 'Symmetric', ha='center', fontsize=13, fontweight='bold')
    ax.text(5.5, max(ratios) * 0.95, 'Stratified', ha='center', fontsize=13, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Proletariat / Bourgeoisie shift ratio', fontsize=12)
    ax.set_title('Figure 4: Who moved more?', fontsize=13, fontweight='bold')
    ax.legend(handles=[
        Patch(facecolor='#D0021B', alpha=0.85, label='Proletariat moves much more (>2×)'),
        Patch(facecolor='#888888', alpha=0.85, label='Roughly equal'),
        Patch(facecolor='#4A90D9', alpha=0.85, label='Bourgeoisie moves more (<0.8×)'),
    ], fontsize=10, loc='upper left')
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout(); plt.savefig('fig4_shift.png', dpi=150, bbox_inches='tight')
    print("  Saved fig4_shift.png")


if __name__ == "__main__":
    print("Run main.py to generate results first, then this file will be called automatically.")

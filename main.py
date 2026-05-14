
import numpy as np
import time

MASTER_SEED = 42

def get_seeds(n_reps=50, master_seed=MASTER_SEED):
    """Generate independent seed sequences from a single master seed."""
    master_rng = np.random.default_rng(master_seed)
    return {
        'C': master_rng.integers(0, 2**31, size=n_reps),
        'W': master_rng.integers(0, 2**31, size=n_reps),
        'X': master_rng.integers(0, 2**31, size=n_reps),
    }



def sample_C_bourg(rng):
    C = np.zeros((3, 3))
    h_col = rng.uniform(0.1, 0.4); w_col = rng.uniform(0.4, 0.7)
    o_col = 1.0 - h_col - w_col
    if o_col < 0.05: o_col = 0.05; h_col = 1.0 - w_col - o_col
    C[0] = [h_col, w_col, o_col]
    neg_o = rng.uniform(-0.5, -0.1); w_self = rng.uniform(0.7, 0.9)
    C[1] = [1.0 - w_self - neg_o, w_self, neg_o]
    w_col2 = rng.uniform(0.4, 0.7); o_self = rng.uniform(0.1, 0.35)
    h_col2 = 1.0 - w_col2 - o_self
    if h_col2 < 0.05: h_col2 = 0.05; o_self = 1.0 - w_col2 - h_col2
    C[2] = [h_col2, w_col2, o_self]
    return C

def sample_C_mid(rng):
    C = np.zeros((3, 3))
    C[0] = rng.dirichlet([3, 1.5, 3])
    C[1] = rng.dirichlet([1.5, 3, 3])
    C[2] = rng.dirichlet([1.5, 1.5, 5])
    return C

def sample_C_prol(rng):
    C = np.zeros((3, 3))
    h_self = rng.uniform(0.5, 0.8); neg_w = rng.uniform(-0.4, -0.2)
    C[0] = [h_self, neg_w, 1.0 - h_self - neg_w]
    C[1] = rng.dirichlet([1.5, 3, 3])
    C[2] = rng.dirichlet([1.5, 3, 3])
    return C

def sample_C_set(rng):
    return [sample_C_bourg(rng), sample_C_mid(rng), sample_C_prol(rng)]



TOPIC_NAMES = ['Health', 'Wealth', 'Opportunity']
CLASS_NAMES = ['Bourgeoisie', 'Middle', 'Proletariat']
INIT_MEANS = {0: [+0.4, +0.7, +0.5], 1: [0.0, 0.0, 0.0], 2: [-0.4, -0.7, -0.5]}

def make_initial_X(N_per_class, means_dict=None, sigma=0.1, seed=None):
    if means_dict is None: means_dict = INIT_MEANS
    rng = np.random.default_rng(seed)
    X = np.zeros((sum(N_per_class), 3)); idx = 0
    for cls, n in enumerate(N_per_class):
        for _ in range(n):
            X[idx] = np.clip(rng.normal(means_dict[cls], sigma), -1.0, 1.0); idx += 1
    return X

def make_W_symmetric(N_total, p=0.05, w_self=0.3, seed=None):
    rng = np.random.default_rng(seed)
    W = np.zeros((N_total, N_total))
    for i in range(N_total):
        for j in range(i+1, N_total):
            if rng.random() < p: W[i,j] = 1; W[j,i] = 1
    for i in range(N_total):
        nb = np.where(W[i] > 0)[0]; W[i] = 0; W[i,i] = w_self
        if len(nb) > 0: W[i, nb] = (1 - w_self) / len(nb)
        else: W[i, i] = 1.0
    return W

def make_W_stratified(N_per_class, p_intra=0.05, k_inter=5, w_self=0.3, seed=None):
    w_intra = {0: 0.7, 1: 0.5, 2: 0.4}
    w_inter = {0: {}, 1: {0: 0.2}, 2: {0: 0.15, 1: 0.15}}
    rng = np.random.default_rng(seed)
    N_total = sum(N_per_class)
    starts = [0]
    for n in N_per_class: starts.append(starts[-1] + n)
    class_of = np.zeros(N_total, dtype=int)
    for cls in range(3): class_of[starts[cls]:starts[cls+1]] = cls
    W = np.zeros((N_total, N_total))
    intra_nb = {i: [] for i in range(N_total)}
    for cls in range(3):
        members = list(range(starts[cls], starts[cls+1]))
        for ii, i in enumerate(members):
            for j in members[ii+1:]:
                if rng.random() < p_intra: intra_nb[i].append(j); intra_nb[j].append(i)
    for i in range(N_total):
        c = class_of[i]; W[i,i] = w_self
        if intra_nb[i]: W[i, intra_nb[i]] = w_intra[c] / len(intra_nb[i])
        else: W[i,i] += w_intra[c]
        for src in w_inter[c]:
            cands = list(range(starts[src], starts[src+1]))
            chosen = rng.choice(cands, size=min(k_inter, len(cands)), replace=False)
            W[i, chosen] = w_inter[c][src] / len(chosen)
    rs = W.sum(axis=1, keepdims=True); rs[rs==0] = 1
    return W / rs

def make_G(suppressed_topic=None):
    if suppressed_topic is None: return np.eye(3)
    discussed = [t for t in range(3) if t != suppressed_topic]
    G = np.zeros((len(discussed), 3))
    for row, t in enumerate(discussed): G[row, t] = 1
    return G

def simulate(X0, W, C_all, G, T_max=400, conv_tol=1e-5, conv_window=10, lam=0.7):
    N, m = X0.shape
    GtG = G.T @ G; I_GtG = np.eye(m) - GtG
    U = X0.copy(); X = X0.copy(); last_diffs = []
    for t in range(T_max):
        X_new = lam * np.einsum('ijk,ik->ij', C_all, X @ I_GtG + W @ X @ GtG) + (1-lam) * U
        diff = np.max(np.abs(X_new - X))
        last_diffs.append(diff)
        if len(last_diffs) > conv_window: last_diffs.pop(0)
        X = X_new
        if len(last_diffs) == conv_window and max(last_diffs) < conv_tol: return X, t+1
    return X, None



def class_means(X, N_per_class):
    starts = [0]
    for n in N_per_class: starts.append(starts[-1] + n)
    return np.array([X[starts[c]:starts[c+1]].mean(axis=0) for c in range(3)])

def inter_class_divergence(X, N_per_class):
    cm = class_means(X, N_per_class)
    return np.array([np.mean([abs(cm[a,k]-cm[b,k]) for a,b in [(0,1),(0,2),(1,2)]]) for k in range(3)])

def class_mean_shift(Xf, X0, N_per_class):
    return np.abs(class_means(Xf, N_per_class) - class_means(X0, N_per_class))



N_PER_CLASS = [333, 333, 334]
N_TOTAL = sum(N_PER_CLASS)
N_REPS = 50
LAM = 0.7

EXPERIMENTS = {
    'E1': ('symmetric', None),  'E2': ('symmetric', 2),
    'E3': ('symmetric', 0),     'E4': ('symmetric', 1),
    'E5': ('stratified', None), 'E6': ('stratified', 2),
    'E7': ('stratified', 0),    'E8': ('stratified', 1),
}

def run_all():
    seeds = get_seeds(n_reps=N_REPS, master_seed=MASTER_SEED)
    print(f"{'='*70}")
    print(f"2x4 EXPERIMENT — N={N_TOTAL}, reps={N_REPS}, lam={LAM}, master_seed={MASTER_SEED}")
    print(f"{'='*70}\n")
    results = {}
    for en, (wt, sp) in EXPERIMENTS.items():
        t0 = time.time(); all_div, all_shift = [], []
        for r in range(N_REPS):
            rng_c = np.random.default_rng(seeds['C'][r])
            C_set = sample_C_set(rng_c)
            C_all = np.zeros((N_TOTAL, 3, 3)); idx = 0
            for cls, n in enumerate(N_PER_CLASS):
                for _ in range(n): C_all[idx] = C_set[cls]; idx += 1
            X0 = make_initial_X(N_PER_CLASS, seed=int(seeds['X'][r]))
            W = make_W_stratified(N_PER_CLASS, seed=int(seeds['W'][r])) if wt == 'stratified' \
                else make_W_symmetric(N_TOTAL, seed=int(seeds['W'][r]))
            G = make_G(suppressed_topic=sp)
            Xf, _ = simulate(X0, W, C_all, G, lam=LAM)
            if np.any(np.abs(Xf) > 10): continue
            all_div.append(inter_class_divergence(Xf, N_PER_CLASS))
            all_shift.append(class_mean_shift(Xf, X0, N_PER_CLASS))
        d = np.array(all_div).mean(axis=0)
        results[en] = {'div_mean': d, 'div_std': np.array(all_div).std(axis=0),
                       'shift_mean': np.array(all_shift).mean(axis=0), 'n_ok': len(all_div)}
        sl = '—' if sp is None else TOPIC_NAMES[sp]
        print(f"  {en} {wt:10s} suppr={sl:12s} | div_mean={d.mean():.3f} ({len(all_div)}/{N_REPS} ok, {time.time()-t0:.1f}s)")
    return results

def print_tables(results):
    e1, e5 = results['E1']['div_mean'], results['E5']['div_mean']
    
    
    print(f"\n{'='*85}")
    print("TABLE 2: Inter-class divergence by experiment")
    print(f"{'='*85}")
    print(f"  {'Exp':>4s} {'Network':>10s} {'Suppressed':>11s} | {'Health':>7s} {'Wealth':>7s} {'Opp':>7s} {'Mean':>7s}")
    print(f"  {'-'*4} {'-'*10} {'-'*11}-+-{'-'*7}-{'-'*7}-{'-'*7}-{'-'*7}")
    for en in EXPERIMENTS:
        wt, sp = EXPERIMENTS[en]
        sl = '—' if sp is None else TOPIC_NAMES[sp]
        d = results[en]['div_mean']
        print(f"  {en:>4s} {wt:>10s} {sl:>11s} | {d[0]:7.3f} {d[1]:7.3f} {d[2]:7.3f} {d.mean():7.3f}")
    
    
    print(f"\n{'='*85}")
    print("TABLE 3: Suppression effect (delta vs baseline, per topic)")
    print(f"{'='*85}")
    print(f"  {'Condition':>22s} | {'ΔHealth':>8s} {'ΔWealth':>8s} {'ΔOpp':>8s} {'ΔMean':>8s}")
    print(f"  {'-'*22}-+-{'-'*8}-{'-'*8}-{'-'*8}-{'-'*8}")
    for exp, base, lab in [('E2','E1','Sym suppr Opp'),('E3','E1','Sym suppr Health'),
                           ('E4','E1','Sym suppr Wealth'),
                           ('E6','E5','Strat suppr Opp'),('E7','E5','Strat suppr Health'),
                           ('E8','E5','Strat suppr Wealth')]:
        delta = results[exp]['div_mean'] - results[base]['div_mean']
        print(f"  {lab:>22s} | {delta[0]:+8.3f} {delta[1]:+8.3f} {delta[2]:+8.3f} {delta.mean():+8.3f}")



def run_robustness():
    """Run all three robustness checks: C-matrix, initial opinions, lambda."""
    seeds = get_seeds(n_reps=N_REPS, master_seed=MASTER_SEED)
    
    
    def _run_one(wt, sp, lam_val, n_pc, means=None, n_reps=None, seeds_=None):
        if n_reps is None: n_reps = N_REPS
        if seeds_ is None: seeds_ = seeds
        if means is None: means = INIT_MEANS
        nt = sum(n_pc)
        divs = []
        for r in range(n_reps):
            rng_c = np.random.default_rng(seeds_['C'][r])
            C_set = sample_C_set(rng_c)
            C_all = np.zeros((nt, 3, 3)); idx = 0
            for cls, n in enumerate(n_pc):
                for _ in range(n): C_all[idx] = C_set[cls]; idx += 1
            X0 = make_initial_X(n_pc, means, seed=int(seeds_['X'][r]))
            W = make_W_stratified(n_pc, seed=int(seeds_['W'][r])) if wt == 'stratified' \
                else make_W_symmetric(nt, seed=int(seeds_['W'][r]))
            G = make_G(suppressed_topic=sp)
            Xf, _ = simulate(X0, W, C_all, G, lam=lam_val)
            if np.any(np.abs(Xf) > 10): continue
            divs.append(inter_class_divergence(Xf, n_pc).mean())
        return np.array(divs)
    
    print(f"\n{'='*85}")
    print("ROBUSTNESS (a): C-matrix variation — per-run consistency")
    print(f"{'='*85}")
    
    per_run = {}
    for en, (wt, sp) in EXPERIMENTS.items():
        per_run[en] = _run_one(wt, sp, LAM, N_PER_CLASS).tolist()
    
    f1_sym = sum(1 for r in range(N_REPS) if all(per_run[e][r] > per_run['E1'][r] for e in ['E2','E3','E4']))
    f1_str = sum(1 for r in range(N_REPS) if all(per_run[e][r] > per_run['E5'][r] for e in ['E6','E7','E8']))
    f2 = sum(1 for r in range(N_REPS) if (per_run['E4'][r]-per_run['E1'][r]) > (per_run['E2'][r]-per_run['E1'][r]))
    f3 = sum(1 for r in range(N_REPS) if
        (per_run['E6'][r]-per_run['E5'][r]) > (per_run['E2'][r]-per_run['E1'][r]) and
        (per_run['E8'][r]-per_run['E5'][r]) < (per_run['E4'][r]-per_run['E1'][r]))
    
    print(f"  Finding 1 (symmetric):  {f1_sym}/{N_REPS} runs")
    print(f"  Finding 1 (stratified): {f1_str}/{N_REPS} runs")
    print(f"  Finding 2 (W > O):      {f2}/{N_REPS} runs")
    print(f"  Finding 3 (direction):  {f3}/{N_REPS} runs")
    
    print(f"\n{'='*85}")
    print("ROBUSTNESS (b): Initial opinion sensitivity")
    print(f"{'='*85}")
    
    N_PC_SMALL = [100, 100, 100]
    NR_SMALL = 20
    seeds_small = get_seeds(n_reps=NR_SMALL, master_seed=MASTER_SEED)
    
    configs = {
        'Original':      {0: [0.4, 0.7, 0.5],  1: [0, 0, 0], 2: [-0.4, -0.7, -0.5]},
        'Milder (0.7x)': {0: [0.28, 0.49, 0.35], 1: [0, 0, 0], 2: [-0.28, -0.49, -0.35]},
        'Extreme (1.3x)':{0: [0.52, 0.91, 0.65], 1: [0, 0, 0], 2: [-0.52, -0.91, -0.65]},
        'Asymmetric':    {0: [0.6, 0.8, 0.3],   1: [0.1, -0.1, 0.2], 2: [-0.3, -0.5, -0.6]},
    }
    
    print(f"  {'Config':>16s} | {'G_opp':>8s} {'G_wealth':>8s} {'W/O ratio':>10s}")
    print(f"  {'-'*50}")
    for name, means in configs.items():
        e1 = _run_one('symmetric', None, LAM, N_PC_SMALL, means, NR_SMALL, seeds_small).mean()
        e2 = _run_one('symmetric', 2, LAM, N_PC_SMALL, means, NR_SMALL, seeds_small).mean()
        e4 = _run_one('symmetric', 1, LAM, N_PC_SMALL, means, NR_SMALL, seeds_small).mean()
        go = e2 - e1; gw = e4 - e1
        ratio = gw / go if abs(go) > 0.001 else 99
        print(f"  {name:>16s} | {go:+8.3f} {gw:+8.3f} {ratio:10.1f}x")
    
    print(f"\n{'='*85}")
    print("ROBUSTNESS (c): Lambda sensitivity")
    print(f"{'='*85}")
    
    lambdas = [0.3, 0.5, 0.7, 0.9]
    print(f"  {'λ':>5s} | {'G_opp_sym':>10s} {'G_w_sym':>10s} {'W/O':>6s} | {'G_opp_str':>10s} {'Opp ratio':>10s}")
    print(f"  {'-'*65}")
    for lv in lambdas:
        e1 = _run_one('symmetric', None, lv, N_PC_SMALL, n_reps=NR_SMALL, seeds_=seeds_small).mean()
        e2 = _run_one('symmetric', 2, lv, N_PC_SMALL, n_reps=NR_SMALL, seeds_=seeds_small).mean()
        e4 = _run_one('symmetric', 1, lv, N_PC_SMALL, n_reps=NR_SMALL, seeds_=seeds_small).mean()
        e5 = _run_one('stratified', None, lv, N_PC_SMALL, n_reps=NR_SMALL, seeds_=seeds_small).mean()
        e6 = _run_one('stratified', 2, lv, N_PC_SMALL, n_reps=NR_SMALL, seeds_=seeds_small).mean()
        gs = e2 - e1; gw = e4 - e1; gr = e6 - e5
        wo = gw / gs if abs(gs) > 0.001 else 99
        oratio = gr / gs if abs(gs) > 0.001 else 99
        print(f"  {lv:5.1f} | {gs:+10.3f} {gw:+10.3f} {wo:6.1f}x | {gr:+10.3f} {oratio:10.1f}x")


if __name__ == "__main__":
    results = run_all()
    print_tables(results)
    print(f"\nRunning robustness tests...")
    run_robustness()
    print(f"\nGenerating figures...")
    from plots import make_all_figures
    make_all_figures(results)
    print(f"\nDone. Master seed: {MASTER_SEED}")

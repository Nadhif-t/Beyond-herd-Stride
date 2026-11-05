"""Experiment driver. python run_all.py --stage {tau,tune,main,frontier,scenarios,global,all} [--quick] [--out outputs] [--calib configs/calib.json]"""
import argparse, json, os, numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from scipy import stats
import sim, stride_analysis as sa

NAME = {('stride', 'targeted'): 'STRIDE', ('stride', 'uniform'): 'STRIDE-uniform', ('stride', 'shuffled'): 'STRIDE-shuffled',
        'poppenalty': 'PopPenalty', 'ucb': 'UCB', 'quota': 'Quota', 'newquota': 'NewQuota', 'expcap': 'ExpCap'}
HYB = {'poppenalty': 0.0, 'ucb': 0.0, 'quota': 0.0, 'newquota': 0.0, 'expcap': 99.0, 'stride': 0.0}   # parameter value that equals Hybrid
GRID = {'stride': [.05, .10, .15, .20, .25, .30, .40], 'poppenalty': [.05, .1, .2, .3, .5], 'ucb': [.05, .1, .2, .4],
        'quota': [.2, .4, .6, .8], 'newquota': [.4, .6, .8, .9, 1.0], 'expcap': [2, 3, 5, 8]}
SCEN = {'Reference': {}, 'Explorers 0%': dict(explorer_share=0), 'Explorers 10%': dict(explorer_share=.1), 'Explorers 40%': dict(explorer_share=.4),
        'Weak herding (kappa=0.5)': dict(kappa=.5), 'Strong herding (kappa=3.5)': dict(kappa=3.5),
        'Abundant capital (30/day)': dict(lam=30), 'Scarce capital (55/day)': dict(lam=55),
        'Weak novelty taste (nu=0.6)': dict(nu=.6), 'Strong novelty taste (nu=1.8)': dict(nu=1.8),
        'Steeper position bias (0.5)': dict(pos_decay=.5), 'Flatter position bias (0.1)': dict(pos_decay=.1),
        'Small loans (median 10)': dict(loan_median=10), 'Large loans (median 25)': dict(loan_median=25),
        'Longer window (W=21)': dict(W=21), 'Shorter window (W=7)': dict(W=7), 'List length K=5': dict(K=5), 'List length K=20': dict(K=20),
        'Fewer first-time (50%)': dict(share_ft=.5), 'More first-time (80%)': dict(share_ft=.8),
        'Weaker track-record (n0=5)': dict(n0=5), 'Stronger track-record (n0=25)': dict(n0=25),
        'No grade signal (0)': dict(grade_signal=0.0), 'Strong grade signal (1.0)': dict(grade_signal=1.0, grade_w=1.0),
        'Creditworthy PD<=10%': dict(cw_pd=.10), 'Creditworthy PD<=20%': dict(cw_pd=.20), 'Benchmark parameters': 'benchmark'}
GLOBAL = dict(kappa=(.5, 3.5), nu=(.6, 1.8), explorer_share=(0, .4), lam=(30, 55), pos_decay=(.1, .5), loan_median=(10, 25),   # [ASSUMED] ranges
              share_ft=(.5, .8), pd_sd=(.1, .5), grade_w=(0, 1.0), n0=(5, 25))

def label(policy, variant):
    return NAME.get((policy, variant)) or NAME.get(policy) or policy.capitalize() if policy != 'hybrid' else 'Hybrid'
def _job(a):
    p, s, pol, prm, var = a; r = sim.run(p, s, pol, prm, var)
    return dict(policy=label(pol, var) if pol != 'hybrid' else 'Hybrid', param=prm, seed=s, **r)
def batch(p, seeds, specs, workers):
    jobs = [(p, s, pol, prm, var) for s in seeds for (pol, prm, var) in specs]
    if workers <= 1: return pd.DataFrame([_job(j) for j in jobs])
    with ProcessPoolExecutor(workers) as ex: return pd.DataFrame(list(ex.map(_job, jobs, chunksize=1)))
def holm(pv):
    o = np.argsort(pv); m = len(pv); adj = np.empty(m); run = 0
    for r, i in enumerate(o): run = max(run, (m-r)*pv[i]); adj[i] = min(1, run)
    return adj
def paired(df, a, b, metric):
    x = df[df.policy == a].sort_values('seed')[metric].values; y = df[df.policy == b].sort_values('seed')[metric].values; d = x-y
    n = len(d); se = d.std(ddof=1)/np.sqrt(n) if n > 1 else np.nan
    return d.mean(), stats.t.ppf(.975, max(n-1, 1))*se, (stats.ttest_1samp(d, 0).pvalue if n > 1 else np.nan)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--stage', default='all'); ap.add_argument('--quick', action='store_true')
    ap.add_argument('--out', default='outputs'); ap.add_argument('--calib', default='configs/calib.json'); ap.add_argument('--workers', type=int, default=os.cpu_count())
    A = ap.parse_args(); os.makedirs(A.out, exist_ok=True); os.makedirs('configs', exist_ok=True)
    calib = json.load(open(A.calib)) if os.path.exists(A.calib) else None
    if calib is None: print('WARNING: no Prosper calibration file -> using the invented fallback loan side (NOT the paper calibration)')
    p0 = sim.Params(calib=calib)
    if calib: p0 = replace(p0, share_ft=calib['share_ft'], loan_logsd=calib['loan_logsd'], lgd=calib['lgd'])
    q = A.quick
    if q: p0 = replace(p0, n_inv=150, lam=8, W=7, T=30, burn=6, eval_lo=6, eval_hi=20)
    R = (lambda a, b: list(range(a, a+(2 if q else b-a+1))))
    tauf = 'configs/tau.json'
    if A.stage in ('tau', 'all'):                                  # calibrate tau so Hybrid funds ~55% of volume (seeds 900-902)
        lo, hi = 0.0, 4.0
        for _ in range(4 if q else 9):
            mid = (lo+hi)/2; v = batch(replace(p0, tau=mid), R(900, 902), [('hybrid', 0, 'targeted')], A.workers).volume.mean()
            lo, hi = (mid, hi) if v > 55 else (lo, mid)
        json.dump(dict(tau=(lo+hi)/2, hybrid_volume=v), open(tauf, 'w')); print('tau', (lo+hi)/2, 'volume', v)
    if os.path.exists(tauf): p0 = replace(p0, tau=json.load(open(tauf))['tau'])
    tuned = 'configs/tuned.json'
    if A.stage in ('tune', 'all'):                                 # rule: max solvent funded s.t. volume >= 97% Hybrid (seeds 101-104)
        seeds = R(101, 104); hyb = batch(p0, seeds, [('hybrid', 0, 'targeted')], A.workers).volume.mean(); best = {}
        for fam, grid in GRID.items():
            d = batch(p0, seeds, [(fam, g, 'targeted') for g in grid], A.workers).groupby('param')[['solvent', 'volume']].mean()
            ok = d[d.volume >= 0.97*hyb]; best[fam] = float(ok.solvent.idxmax()) if len(ok) else HYB[fam]
            d.to_csv(f'{A.out}/tuning_{fam}.csv')
        json.dump(best, open(tuned, 'w')); print('tuned', best)
    T = json.load(open(tuned)) if os.path.exists(tuned) else {'stride': .2, 'poppenalty': .1, 'ucb': .1, 'quota': .6, 'newquota': .8, 'expcap': 5}
    if A.stage in ('main', 'all'):
        specs = [(x, 0, 'targeted') for x in ('random', 'popularity', 'cf', 'content', 'hybrid')] + [(f, T[f], 'targeted') for f in GRID] + \
                [('stride', T['stride'], 'uniform'), ('stride', T['stride'], 'shuffled')]
        df = batch(p0, R(1, 30), specs, A.workers); df.to_csv(f'{A.out}/main.csv', index=False); rows = []
        for pol in df.policy.unique():
            r = dict(policy=pol)
            for m in ('npsr', 'solvent', 'volume', 'incumbents', 'net_return', 'loss_rate', 'idle'):
                d, h, pv = paired(df, pol, 'Hybrid', m); r[m] = df[df.policy == pol][m].mean(); r['d_'+m] = d; r['ci_'+m] = h; r['p_'+m] = pv
            rows.append(r)
        t = pd.DataFrame(rows); pcols = [c for c in t.columns if c.startswith('p_')]
        allp = t[pcols].values.flatten(); ok = ~np.isnan(allp); adj = np.full(allp.shape, np.nan); adj[ok] = holm(allp[ok])   # Holm over policies x metrics
        t[[c+'_holm' for c in pcols]] = adj.reshape(t[pcols].shape); t.to_csv(f'{A.out}/table3_4.csv', index=False)
        vr = {pol: sa.volume_margin(df[df.policy == pol].sort_values('seed').volume.values, df[df.policy == 'Hybrid'].sort_values('seed').volume.values)
              for pol in df.policy.unique() if pol != 'Hybrid'}
        pd.DataFrame(vr).T.to_csv(f'{A.out}/volume_margins_eval.csv'); print(t[['policy', 'npsr', 'volume', 'incumbents']])
    if A.stage in ('frontier', 'all'):
        specs = [('stride', g, 'targeted') for g in [0]+GRID['stride']] + [('stride', .2, 'uniform')] + \
                [(f, g, 'targeted') for f in ('poppenalty', 'ucb', 'newquota', 'expcap') for g in [HYB[f]]+GRID[f]]
        df = batch(p0, R(201, 216), specs, A.workers); df.to_csv(f'{A.out}/frontier.csv', index=False); rows = []
        for w3 in (.10, .15, .20, .25, .30):
            for comp in ('PopPenalty', 'NewQuota'):
                for metric in ('npsr', 'solvent', 'incumbents', 'idle'):
                    r = sa.equal_volume_diff(df.rename(columns={}), w3, comp, metric, B=200 if q else 1000, seed=0)
                    rows.append(dict(w3=w3, comparator=comp, metric=metric, **r, verdict=(sa.equivalence_verdict(r['lo'], r['hi']) if r.get('status') == 'ok' else 'unresolved')))
        pd.DataFrame(rows).to_csv(f'{A.out}/table5_6.csv', index=False)
    if A.stage in ('scenarios', 'all'):
        rows = []
        for name, mod in SCEN.items():
            p = replace(p0, calib=sim.benchmark_calib(), lgd=.6) if mod == 'benchmark' else replace(p0, **mod)
            df = batch(p, R(301, 305), [('hybrid', 0, 'targeted'), ('stride', T['stride'], 'targeted'), ('stride', T['stride'], 'uniform'), ('poppenalty', T['poppenalty'], 'targeted')], A.workers)
            df = df.assign(policy=df.policy.replace({'STRIDE-uniform': 'U'}))
            s1 = paired(df, 'STRIDE', 'Hybrid', 'npsr'); s2 = paired(df, 'STRIDE', 'Hybrid', 'volume'); s3 = paired(df, 'PopPenalty', 'Hybrid', 'npsr'); s4 = paired(df, 'STRIDE', 'U', 'npsr')
            rows.append(dict(scenario=name, hybrid_npsr=df[df.policy == 'Hybrid'].npsr.mean(), d_npsr=s1[0], d_vol=s2[0], d_pop=s3[0], d_target=s4[0], p_target=s4[2]))
        t = pd.DataFrame(rows); t['holm_p'] = holm(t.p_target.fillna(1).values); t.to_csv(f'{A.out}/table7_scenarios.csv', index=False)
    if A.stage in ('global', 'all'):
        rng = np.random.default_rng(2026); rows = []
        for d in range(4 if q else 24):
            prm = {k: float(rng.uniform(*v)) for k, v in GLOBAL.items()}; p = replace(p0, **prm)
            specs = [('hybrid', 0, 'targeted'), ('stride', .2, 'targeted'), ('stride', .2, 'uniform')] + [('poppenalty', l, 'targeted') for l in (.1, .2, .3)]
            df = batch(p, [400+10*d+j for j in range(2 if q else 3)], specs, A.workers); m = df.groupby(['policy', 'param']).mean(numeric_only=True).reset_index()
            cur = pd.concat([m[m.policy == 'Hybrid'].assign(param=0.0), m[m.policy == 'PopPenalty']]).sort_values('volume')
            st = m[(m.policy == 'STRIDE')].iloc[0]; un = m[m.policy == 'STRIDE-uniform'].iloc[0]
            val, br, status = sa.interp_at_volume(cur, st.volume, 'npsr')
            rows.append(dict(draw=d, **prm, gap=st.npsr-val if status == 'ok' else np.nan, target=st.npsr-un.npsr, gain=st.npsr-m[m.policy == 'Hybrid'].npsr.iloc[0]))
        g = pd.DataFrame(rows); g.to_csv(f'{A.out}/global.csv', index=False)
        print({k: round(stats.spearmanr(g[k], g.gap, nan_policy='omit').statistic, 2) for k in GLOBAL})

if __name__ == '__main__': main()

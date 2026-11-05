"""
Post-processing utilities for the STRIDE revision (review-95).
They compute, from YOUR per-replication outputs and the Prosper file, the
quantities that appear in [TO COMPLETE] / [AUTHORS TO CONFIRM] in the manuscript.
They do NOT contain the simulator. Expected input format is described per function.
"""
import numpy as np, pandas as pd
from scipy import stats

# ---------------------------------------------------------------- 5.1  (R1.5)
def volume_margin(vol_policy, vol_hybrid, floor=0.97, B=2000, seed=0):
    """vol_* : arrays indexed by the SAME seeds (paired). Returns mean ratio,
    t-interval, bootstrap one-sided 95% lower bound, and margin to the floor.
    Run once on tuning seeds (101-104) and once on evaluation seeds (1-30)."""
    r = np.asarray(vol_policy) / np.asarray(vol_hybrid)
    n = len(r); m = r.mean()
    se = r.std(ddof=1) / np.sqrt(n)
    tcrit = stats.t.ppf(0.975, n - 1)
    rng = np.random.default_rng(seed)
    boot = np.array([r[rng.integers(0, n, n)].mean() for _ in range(B)])
    return dict(n=n, ratio=m, t_lo=m - tcrit*se, t_hi=m + tcrit*se,
                boot_lower95_onesided=np.percentile(boot, 5),
                margin_to_floor=m - floor, floor_ok=np.percentile(boot, 5) >= floor)

def select_intensity(grid_df, floor=0.97):
    """grid_df columns: param, ratio_lower95 (validation seeds, separate from tuning
    and evaluation). Returns the largest intensity whose lower bound >= floor."""
    ok = grid_df[grid_df.ratio_lower95 >= floor]
    return None if ok.empty else ok.param.max()

# ---------------------------------------------------------------- 5.3  (R1.4)
def _curve(df, policy, metric):
    """df columns: policy, param, seed, volume, <metric>. Mean curve by param."""
    g = df[df.policy == policy].groupby('param')[['volume', metric]].mean().reset_index()
    return g.sort_values('volume').reset_index(drop=True)

def interp_at_volume(curve, v_target, metric):
    """Linear interpolation of the mean curve at v_target.
    Returns (value, bracket, status). If the target is outside the curve range, or the
    curve crosses v_target more than once (non-monotone), status is not 'ok' and the
    comparison must be reported as unresolved (rule stated in Section 5.3)."""
    v = curve.volume.values; y = curve[metric].values; p = curve.param.values
    hits = [i for i in range(len(v)-1) if min(v[i], v[i+1]) <= v_target <= max(v[i], v[i+1]) and v[i] != v[i+1]]
    if not hits: return np.nan, None, 'outside_range'
    if len(hits) > 1: return np.nan, None, 'multiple_crossings'
    i = hits[0]; w = (v_target - v[i]) / (v[i+1] - v[i])
    return y[i] + w*(y[i+1]-y[i]), (p[i], p[i+1], v[i], v[i+1]), 'ok'

def equal_volume_diff(df, w3, comparator, metric='npsr', B=1000, seed=0):
    """STRIDE (param==w3) minus comparator at equal mean funded volume.
    Replication indices (seeds) are resampled JOINTLY for all policies/settings and the
    interpolation is REPEATED inside each resample. Returns point estimate, percentile
    95% interval, point-estimate interpolation bracket and the volume mismatch
    (|target - nearest comparator volume|)."""
    seeds = np.sort(df.seed.unique()); rng = np.random.default_rng(seed)
    def stat(sub):
        s = sub[(sub.policy == 'STRIDE') & (np.isclose(sub.param, w3))]
        if s.empty: return np.nan, None, 'no_stride'
        vt = s.volume.mean(); val, br, st = interp_at_volume(_curve(sub, comparator, metric), vt, metric)
        return s[metric].mean() - val, br, st, vt
    est = stat(df)
    out = dict(est=est[0], bracket=est[1], status=est[2])
    if est[2] != 'ok': return out
    cmpv = _curve(df, comparator, metric).volume.values
    out['target_volume'] = est[3]; out['volume_mismatch'] = float(np.min(np.abs(cmpv - est[3])))
    boots = []
    for _ in range(B):
        draw = rng.choice(seeds, len(seeds), replace=True)
        sub = pd.concat([df[df.seed == s] for s in draw], ignore_index=True)
        r = stat(sub)
        if r[2] == 'ok': boots.append(r[0])
    out['n_valid_boot'] = len(boots)
    out['lo'], out['hi'] = np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan)
    return out

def equivalence_verdict(lo, hi, delta=2.0):
    """Rule of Section 5.3 / Table 6."""
    if lo >= -delta and hi <= delta:
        return 'equivalent' if lo <= 0 <= hi else 'within margin, detectably non-zero'
    if hi < -delta or lo > delta: return 'different'
    return 'unresolved'

def power_mde(half_width, power=0.80, alpha=0.05):
    """Approximate minimum detectable difference from a reported interval half-width."""
    return half_width * (stats.norm.ppf(1-alpha/2) + stats.norm.ppf(power)) / stats.norm.ppf(1-alpha/2)

# ---------------------------------------------------------------- 4.2  (R1.6)
def calibration_cells(loans, B=2000, seed=0):
    """loans columns: first_time (bool), grade, default (0/1) on the 8,676 resolved loans.
    Returns cell counts and default rates with Wilson 95% intervals, and the
    first-time/repeat default ratio with a loan-level percentile bootstrap interval."""
    rows = []
    for (ft, g), s in loans.groupby(['first_time', 'grade']):
        n, k = len(s), int(s.default.sum())
        lo, hi = stats.binomtest(k, n).proportion_ci(0.95, method='wilson')
        rows.append(dict(first_time=ft, grade=g, n=n, defaults=k, rate=k/n, wilson_lo=lo, wilson_hi=hi))
    cells = pd.DataFrame(rows)
    rng = np.random.default_rng(seed); a = loans[loans.first_time].default.values; b = loans[~loans.first_time].default.values
    ratio = a.mean()/b.mean()
    bs = [rng.choice(a, len(a)).mean()/rng.choice(b, len(b)).mean() for _ in range(B)]
    return cells, dict(ratio=ratio, lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5))

def lognormal_pd(p, sd=0.25, n=200000, seed=0):
    """Latent PD from a cell rate p. Mean-preserving form: PD = p*exp(sd*z - sd^2/2), then
    clipped at 1. Returns the share of draws clipped and the mean before/after clipping,
    i.e. the answer to 'does clipping change the calibrated moments'.
    Replace by YOUR actual transformation if it differs."""
    z = np.random.default_rng(seed).standard_normal(n)
    raw = p*np.exp(sd*z - sd**2/2)
    clipped = np.minimum(raw, 1.0)
    return dict(p=p, mean_raw=raw.mean(), mean_clipped=clipped.mean(), share_clipped=(raw > 1).mean())

# ---------------------------------------------------------------- 5.2  (R1.7)
def newness_overlap(listings, age_cut=5):
    """listings columns: age_at_first_exposure (days), first_time (bool), exposure (impressions).
    Returns the 2x2 overlap (counts, row shares) and mean exposure by cell."""
    L = listings.assign(young=listings.age_at_first_exposure <= age_cut)
    counts = pd.crosstab(L.first_time, L.young)
    shares = counts.div(counts.sum(axis=1), axis=0)
    expo = L.groupby(['first_time', 'young']).exposure.mean().unstack()
    return counts, shares, expo

def exposure_change(expo_policy, expo_hybrid):
    """Both: Series of mean exposure indexed by (first_time, young) -> relative change."""
    return expo_policy/expo_hybrid - 1

# ---------------------------------------------------------------- 6.4 / R1.9
def oracle_auc(score, zeta, top_q=0.75):
    """AUC of a per-investor score for recovering true explorers (zeta in top quartile);
    use score=StrangerScore for the observed AUC (0.59) and score=zeta for the oracle control."""
    from sklearn.metrics import roc_auc_score
    y = zeta >= np.quantile(zeta, top_q)
    return roc_auc_score(y, score), stats.spearmanr(score, zeta).statistic

# ---------------------------------------------------------------- self-test on synthetic data
if __name__ == '__main__':
    rng = np.random.default_rng(1); rows = []
    for s in range(1, 17):
        for pol in ['STRIDE', 'PopPenalty']:
            for p in [0.0, 0.1, 0.2, 0.3] if pol == 'STRIDE' else [0.0, 0.1, 0.2, 0.3, 0.4]:
                v = 55.3 - 9*p + rng.normal(0, .3); rows.append(dict(policy=pol, param=p, seed=s, volume=v, npsr=50.8+60*p - 40*p*p + rng.normal(0, .5)))
    df = pd.DataFrame(rows)
    r = equal_volume_diff(df, 0.2, 'PopPenalty', B=100); print(r, equivalence_verdict(r['lo'], r['hi']))
    print(volume_margin(rng.normal(53.5, .7, 30), rng.normal(55.3, .7, 30)))
    print(lognormal_pd(0.184)); print('MDE', power_mde(1.05))

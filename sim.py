"""Agent-based lending marketplace + recommendation policies (reconstruction from the paper's description).
Every quantity the paper does not state is an ASSUMPTION, tagged [ASSUMED] below and listed in README_RECONSTRUCTION.md."""
import numpy as np
from dataclasses import dataclass
from scipy.sparse.linalg import svds

GRADES = ['AA', 'A', 'B', 'C', 'D', 'E', 'HR']
def sig(x): return 1/(1+np.exp(-x))
def logit(q): q = np.clip(q, 1e-3, 1-1e-3); return np.log(q/(1-q))

def fallback_calib():
    """[ASSUMED] placeholder loan side used ONLY when no Prosper calibration file exists.
    Default-by-grade follows Table 2 (4/9/12/15/22/26/25 %); mixes and rates are invented."""
    base = [.04, .09, .12, .15, .22, .26, .25]
    return dict(
        grade_mix=dict(ft=[.01, .05, .10, .22, .26, .16, .20], rep=[.04, .13, .17, .24, .20, .10, .12]),
        default=dict(ft=[x*1.1 for x in base], rep=[x*0.9 for x in base]),
        rate=dict(ft=[.07, .10, .14, .18, .22, .26, .29], rep=[.07, .10, .14, .18, .22, .26, .29]),
        share_ft=0.651, loan_logsd=0.755, lgd=0.63)

def benchmark_calib():
    """Original SME-style parameters: PD 7% (first-time) and 4.5% (repeat), lgd 0.6; rate 12% [ASSUMED]."""
    return dict(grade_mix=dict(ft=[1/7]*7, rep=[1/7]*7), default=dict(ft=[.07]*7, rep=[.045]*7),
                rate=dict(ft=[.12]*7, rep=[.12]*7), share_ft=0.651, loan_logsd=0.755, lgd=0.6)

@dataclass
class Params:
    n_inv: int = 1200; lam: float = 42; W: int = 14; T: int = 80; burn: int = 20
    eval_lo: int = 20; eval_hi: int = 63
    K: int = 10; tau: float = 1.18; kappa: float = 2.0; nu: float = 1.2; pos_decay: float = 0.25
    explorer_share: float = 0.25          # [ASSUMED]
    grade_w: float = 0.5; grade_signal: float = 1.0; grade_noise: float = 0.4
    n0: float = 15.0                      # track-record strength [ASSUMED default]
    share_ft: float = 0.651; pd_sd: float = 0.25; lgd: float = 0.63
    loan_median: float = 15.0; loan_logsd: float = 0.755; loan_clip: tuple = (3.0, 60.0)   # [ASSUMED]
    ticket_unit: float = 1.0; budget: int = 2; n_sectors: int = 8                        # [ASSUMED]
    cw_pd: float = 0.15; tau_age: int = 5; svd_rank: int = 12; svd_every: int = 3
    shrink_k: float = 3.0; tail_q: float = 0.70
    calib: dict = None

def make_world(p, s):
    c = p.calib or fallback_calib(); rng = np.random.default_rng([s, 0])
    U, S = p.n_inv, p.n_sectors
    aff = rng.uniform(0, 1, (U, S))                                   # [ASSUMED] sector affinities U(0,1)
    expl = rng.random(U) < p.explorer_share
    zeta = np.where(expl, rng.uniform(.7, 1, U), rng.uniform(0, .3, U))  # [ASSUMED] latent novelty taste
    counts = rng.poisson(p.lam, p.T); N = int(counts.sum()); day = np.repeat(np.arange(p.T), counts)
    ft = rng.random(N) < p.share_ft
    grade = np.zeros(N, int); dd = np.zeros(N); rate = np.zeros(N)
    for typ, key in ((True, 'ft'), (False, 'rep')):
        idx = np.where(ft == typ)[0]; mix = np.array(c['grade_mix'][key], float); mix /= mix.sum()
        grade[idx] = rng.choice(7, len(idx), p=mix)
        dd[idx] = np.array(c['default'][key])[grade[idx]]; rate[idx] = np.array(c['rate'][key])[grade[idx]]
    z = rng.standard_normal(N)
    pdv = np.clip(dd*np.exp(p.pd_sd*z - p.pd_sd**2/2), 0, 0.95)      # mean-preserving lognormal, clipped [ASSUMED]
    amount = np.clip(p.loan_median*np.exp(p.loan_logsd*rng.standard_normal(N)), *p.loan_clip)
    m = np.maximum(2, np.round(amount/p.ticket_unit)).astype(int)
    return dict(aff=aff, zeta=zeta, day=day, ft=ft, pd=pdv, rate=rate, amount=amount, m=m,
                sector=rng.integers(0, S, N), q=np.clip(1-dd/0.3, 0.05, 0.95),
                rho=rate*(1-pdv) - p.lgd*pdv)

def mm(X):
    lo = X.min(1, keepdims=True); hi = X.max(1, keepdims=True); r = hi-lo
    return np.where(r > 1e-12, (X-lo)/np.where(r > 1e-12, r, 1), 0.0)   # constant vector -> 0 [ASSUMED]

def stranger(p, hist, tot, ycnt, sap, variant, rng):
    S = hist.shape[1]; has = tot > 0; dist = hist/np.maximum(tot, 1)[:, None]
    ent = -(np.where(dist > 0, dist*np.log(np.where(dist > 0, dist, 1)), 0)).sum(1)/np.log(S)
    out = 0
    for x in (ent, ycnt/np.maximum(tot, 1), sap/np.maximum(tot, 1)):
        prior = x[has].mean() if has.any() else 0.5                   # population prior = current mean [ASSUMED]
        out = out + (x*tot + prior*p.shrink_k)/(tot + p.shrink_k)
    ss = out/3
    if variant == 'uniform': ss = np.full_like(ss, ss.mean())
    elif variant == 'shuffled': ss = rng.permutation(ss)
    return ss

def build_lists(score, valid, K, group=None, q=0.0):
    sc = np.where(valid, score, -np.inf)
    order = np.argsort(-sc, axis=1, kind='stable'); v = np.take_along_axis(valid, order, 1)
    sel = v & (np.cumsum(v, 1) <= K)
    if group is not None and q > 0:
        r = int(round(q*K)); g = group[order]
        a = v & g & (np.cumsum(v & g, 1) <= r)
        rest = v & ~a
        b = rest & (np.cumsum(rest, 1) <= (K - a.sum(1, keepdims=True)))
        sel = a | b                                                   # shortage filled by best remaining; order = base score
    pos = np.cumsum(sel, 1) - 1; L = np.full((score.shape[0], K), -1); rr, cc = np.where(sel)
    L[rr, pos[rr, cc]] = order[rr, cc]
    return L

def run(p, s, policy='hybrid', param=0.0, variant='targeted'):
    w = make_world(p, s); U = p.n_inv; N = len(w['day']); T = p.T; K = p.K
    committed = np.zeros((U, N), bool); nc = np.zeros(N, int); funded = np.zeros(N, bool)
    fday = np.full(N, -1); failday = np.full(N, -1); csum = np.zeros(N); impr = np.zeros(N)
    hist = np.zeros((U, p.n_sectors)); tot = np.zeros(U); ycnt = np.zeros(U); sap = np.zeros(U); Uk = None
    ftw = 50.0/(50.0 + p.n0)                                          # track-record discount on first-time grade term [ASSUMED]
    for t in range(T):
        rng = np.random.default_rng([s, 1000+t])
        failday[(w['day']+p.W == t) & ~funded] = t
        op = np.where((w['day'] <= t) & (t < w['day']+p.W) & ~funded)[0]
        if len(op) == 0: continue
        n = nc[op].astype(float); pop = np.minimum(1, np.log1p(n)/np.log(61)); young = (t-w['day'][op]) <= p.tau_age
        pol, prm = (policy, param) if t >= p.burn else ('hybrid', 0.0)
        if t % p.svd_every == 0 and t > 0:
            created = int((w['day'] <= t).sum()); M = committed[:, :created].astype(np.float32)
            if M.sum() > 0:
                k = min(p.svd_rank, min(M.shape)-1)
                u_, _, _ = svds(M, k=k, v0=np.random.default_rng([s, 5000+t]).standard_normal(min(M.shape))); Uk = u_
        cf = np.zeros((U, len(op)))
        if Uk is not None: cf = Uk @ (Uk.T @ committed[:, op].astype(np.float32))   # fold-in of current investors
        cont = hist[:, w['sector'][op]]/np.maximum(tot, 1)[:, None] + 0.5*w['q'][op][None, :]
        cfn, contn = mm(cf), mm(cont); hyb = 0.5*cfn + 0.5*contn
        group = None; qshare = 0.0
        if pol == 'hybrid': sc = hyb
        elif pol == 'random': sc = rng.random(hyb.shape)
        elif pol == 'popularity': sc = np.tile(n, (U, 1))
        elif pol == 'cf': sc = cfn
        elif pol == 'content': sc = contn
        elif pol == 'poppenalty': sc = hyb - prm*pop[None, :]
        elif pol == 'ucb': sc = hyb + prm/np.sqrt(1+n)[None, :]
        elif pol == 'quota': sc = hyb; group = young; qshare = prm
        elif pol == 'newquota': sc = hyb; group = w['ft'][op]; qshare = prm
        elif pol == 'expcap': sc = hyb
        elif pol == 'stride':
            ss = stranger(p, hist, tot, ycnt, sap, variant, np.random.default_rng([s, 3000+t]))
            tail = n <= np.quantile(n, p.tail_q); elig = (young | tail)
            w1 = (1-prm)/2; sc = w1*cfn + w1*contn + prm*ss[:, None]*((1-pop)*elig)[None, :]
        else: raise ValueError(pol)
        valid = ~committed[:, op]
        if pol == 'expcap':
            lists = np.full((U, K), -1); today = np.zeros(len(op)); capv = prm*U*K/len(op)
            for chunk in np.array_split(np.random.default_rng([s, 7000+t]).permutation(U), 8):
                L = build_lists(sc[chunk], valid[chunk] & (today < capv)[None, :], K)
                lists[chunk] = L; today += np.bincount(L[L >= 0], minlength=len(op))
        else:
            lists = build_lists(sc, valid, K, group, qshare)
        impr[op] += np.bincount(lists[lists >= 0], minlength=len(op))
        budget = np.full(U, p.budget)
        for k in range(K):
            idx = lists[:, k]; ok = (idx >= 0) & (budget > 0)
            ok &= np.where(idx >= 0, nc[op[np.maximum(idx, 0)]] < w['m'][op[np.maximum(idx, 0)]], False)
            us = np.where(ok)[0]
            if len(us) == 0: continue
            us = us[rng.permutation(len(us))]; li = idx[us]; gl = op[li]
            perc = sig(logit(w['q'][gl]) + p.grade_noise*rng.standard_normal(len(us)))*p.grade_signal
            perc = np.where(w['ft'][gl], perc*ftw, perc)
            zt = w['zeta'][us]
            u = w['aff'][us, w['sector'][gl]] + p.grade_w*perc + p.kappa*(1-zt)*pop[li] + p.nu*zt*young[li] - p.tau
            take = rng.random(len(us)) < sig(u)/(1 + p.pos_decay*k)
            us, gl, li = us[take], gl[take], li[take]
            if len(us) == 0: continue
            o = np.argsort(gl, kind='stable'); us, gl, li = us[o], gl[o], li[o]
            first = np.r_[0, np.where(np.diff(gl) != 0)[0]+1]; rank = np.arange(len(gl)) - np.repeat(first, np.diff(np.r_[first, len(gl)]))
            acc = nc[gl] + rank < w['m'][gl]; us, gl, li = us[acc], gl[acc], li[acc]
            committed[us, gl] = True; np.add.at(nc, gl, 1); csum[gl] += t; budget[us] -= 1
            np.add.at(hist, (us, w['sector'][gl]), 1); tot[us] += 1; ycnt[us] += young[li]; sap[us] += 1-pop[li]
            done = (nc >= w['m']) & ~funded; fday[done] = t; funded |= done
    return metrics(p, w, funded, fday, failday, nc, csum, impr)

def gini(x):
    x = np.sort(x.astype(float)); n = len(x); c = x.cumsum()
    return 0.0 if c[-1] == 0 else (n+1-2*(c.sum()/c[-1]))/n

def metrics(p, w, funded, fday, failday, nc, csum, impr):
    ev = (w['day'] >= p.eval_lo) & (w['day'] <= p.eval_hi); ft = w['ft']; a = w['amount']
    res = np.where(funded, fday, np.where(failday >= 0, failday, p.T)); capd = nc*res - csum
    fe = ev & funded; cw = ev & ft & (w['pd'] <= p.cw_pd)
    return dict(npsr=100*funded[ev & ft].mean(), solvent=100*a[cw & funded].sum()/max(a[cw].sum(), 1e-9),
                volume=100*a[fe].sum()/a[ev].sum(), incumbents=100*funded[ev & ~ft].mean(),
                net_return=100*(w['rho']*a)[fe].sum()/max(a[fe].sum(), 1e-9),
                loss_rate=100*(w['pd']*p.lgd*a)[fe].sum()/max(a[fe].sum(), 1e-9),
                idle=100*capd[ev & ~funded].sum()/max(capd[ev].sum(), 1e-9), gini=gini(impr[ev]))

"""Loan-side calibration on the Prosper file (Section 4.1-4.2). Usage: python calibrate.py data/prosperLoanData.csv configs/calib.json"""
import sys, json, hashlib, numpy as np, pandas as pd
from stride_analysis import calibration_cells
G = ['AA', 'A', 'B', 'C', 'D', 'E', 'HR']

def calibrate(path, out):
    sha = hashlib.sha256(open(path, 'rb').read()).hexdigest()
    df = pd.read_csv(path, usecols=['ListingCreationDate', 'Term', 'ProsperRating (Alpha)', 'LoanStatus',
                                    'TotalProsperLoans', 'BorrowerRate', 'LoanOriginalAmount'], parse_dates=['ListingCreationDate'])
    df = df[(df.Term == 36) & df['ProsperRating (Alpha)'].notna() & df.LoanStatus.isin(['Completed', 'Chargedoff', 'Defaulted'])
            & (df.ListingCreationDate >= '2009-07-01') & (df.ListingCreationDate < '2011-03-01')].copy()
    df['first_time'] = df.TotalProsperLoans.isna(); df['default'] = df.LoanStatus.isin(['Chargedoff', 'Defaulted']).astype(int)
    df['grade'] = df['ProsperRating (Alpha)']
    cells, ratio = calibration_cells(df[['first_time', 'grade', 'default']])
    res = dict(n=len(df), n_ft=int(df.first_time.sum()), sha256=sha, share_ft=float(df.first_time.mean()), ratio=ratio,
               loan_logsd=float(np.log(df.LoanOriginalAmount).std()), lgd=0.63, grade_mix={}, default={}, rate={})
    for key, flag in (('ft', True), ('rep', False)):
        s = df[df.first_time == flag]
        res['grade_mix'][key] = [float((s.grade == g).mean()) for g in G]
        res['default'][key] = [float(s[s.grade == g].default.mean()) for g in G]
        res['rate'][key] = [float(s[s.grade == g].BorrowerRate.mean()) for g in G]
    cells.to_csv(out.replace('.json', '_cells.csv'), index=False); json.dump(res, open(out, 'w'), indent=1)
    print({k: res[k] for k in ('n', 'n_ft', 'share_ft', 'ratio', 'sha256')})

if __name__ == '__main__': calibrate(sys.argv[1], sys.argv[2])

#!/usr/bin/env python3
"""
SYNTHETIC Lassa fever line-list generator  (NOT REAL SORMAS / NCDC DATA)
=========================================================================
Purpose : build and test a surveillance pipeline (aggregation -> features -> ASVM)
          when the real SORMAS .sav file is unavailable.
Mimics  : the *structure* described in the dissertation
          - 20,062 case records, 2018-01 .. 2021-12 (48 months), 774 LGAs, 37 states (36 + FCT)
          - aggregated to state-month; features lag-1, lag-2, roll-3, trend, hotspot, sin/cos month
          - outbreak label: confirmed > state's 70th percentile AND confirmed >= 3  (~5.7 % of rows)
It does NOT reproduce real epidemiology. State shares, seasonality, surges, ages, outcomes are
stylised assumptions. Never report results from this file as results on real data.

Usage:  python generate_synthetic_lassa.py --out ./synthetic_lassa_data
"""
import argparse, os
import numpy as np
import pandas as pd

MONTHS = pd.period_range("2018-01", "2021-12", freq="M")          # 48 months
TOTAL_RECORDS = 20_062
TARGET_EVAL_POS = 82            # dissertation: 82 outbreaks among 1,447 evaluation rows
WARMUP_FRAC = 0.15

# state: number of LGAs (sums to 774)
STATES = {
 "Abia":17,"Adamawa":21,"Akwa Ibom":31,"Anambra":21,"Bauchi":20,"Bayelsa":8,"Benue":23,"Borno":27,
 "Cross River":18,"Delta":25,"Ebonyi":13,"Edo":18,"Ekiti":16,"Enugu":17,"FCT":6,"Gombe":11,"Imo":27,
 "Jigawa":27,"Kaduna":23,"Kano":44,"Katsina":34,"Kebbi":21,"Kogi":21,"Kwara":16,"Lagos":20,
 "Nasarawa":13,"Niger":25,"Ogun":20,"Ondo":18,"Osun":30,"Oyo":33,"Plateau":17,"Rivers":23,
 "Sokoto":23,"Taraba":16,"Yobe":17,"Zamfara":14}
assert len(STATES) == 37 and sum(STATES.values()) == 774, (len(STATES), sum(STATES.values()))

# stylised share of confirmed cases (endemic-state concentration); remainder spread thinly
ENDEMIC = {"Edo":.32,"Ondo":.26,"Ebonyi":.11,"Bauchi":.10,"Taraba":.06,"Nasarawa":.02,"Plateau":.015,
           "Benue":.005,"Kogi":.005,"Delta":.005,"Kaduna":.003,"Gombe":.003}
rest = [s for s in STATES if s not in ENDEMIC]
SHARE = dict(ENDEMIC)
for s in rest:
    SHARE[s] = (1 - sum(ENDEMIC.values())) / len(rest)

YEAR_FACTOR = {2018: 0.85, 2019: 1.0, 2020: 1.45, 2021: 0.70}    # stylised year-to-year variation
CONFIRMED_TOTAL_TARGET = 3100                                    # stylised national confirmed total


def simulate_confirmed(rng):
    """Seasonal, autocorrelated, over-dispersed monthly confirmed counts per state."""
    nm = len(MONTHS)
    mon = np.array([p.month for p in MONTHS]); yr = np.array([p.year for p in MONTHS])
    season = np.exp(1.0 * np.cos(2 * np.pi * (mon - 2) / 12.0))   # peak ~Feb (dry season), stylised
    season /= season.mean()
    yf = np.array([YEAR_FACTOR[y] for y in yr]); yf /= yf.mean()
    monthly_total = CONFIRMED_TOTAL_TARGET / nm
    out = {}
    for s in STATES:
        u = np.zeros(nm); eps = rng.normal(0, 0.45, nm)
        for t in range(1, nm):
            u[t] = 0.6 * u[t - 1] + eps[t]
        surge = np.ones(nm)
        if s in ENDEMIC:                                           # occasional multi-month surges
            t = 0
            while t < nm:
                if rng.random() < 0.05:
                    d = rng.integers(1, 4); surge[t:t + d] *= rng.uniform(1.8, 3.2); t += d
                t += 1
        lam = monthly_total * SHARE[s] * season * yf * np.exp(u - 0.5 * 0.45**2) * surge
        k = 1.5                                                    # NB dispersion
        out[s] = rng.poisson(rng.gamma(k, lam / k + 1e-12))
    return pd.DataFrame(out, index=MONTHS)


def build_state_month(conf):
    rows = []
    for s in STATES:
        c = conf[s].values.astype(float)
        q70 = np.percentile(c, 70)                                 # state-specific historical 70th pct
        for t, p in enumerate(MONTHS):
            rows.append((s, p, int(c[t]), q70))
    df = pd.DataFrame(rows, columns=["state", "period", "confirmed", "q70"])
    df["outbreak"] = ((df.confirmed > df.q70) & (df.confirmed >= 3)).astype(int)
    return df


def add_features(df):
    df = df.sort_values(["state", "period"]).copy()
    g = df.groupby("state")["confirmed"]
    df["lag1"] = g.shift(1); df["lag2"] = g.shift(2)
    # roll3: mean of up to the three PRIOR months (no leakage), uses available months only
    df["roll3"] = g.transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    df["trend"] = df["lag1"] - df["lag2"]
    # hotspot: expanding mean of prior months is in the top quartile of states that month
    df["_cum"] = g.transform(lambda x: x.shift(1).expanding().mean())
    thr = df.groupby("period")["_cum"].transform(lambda x: x.quantile(0.75))
    df["hotspot"] = (df["_cum"] >= thr).astype(int)
    m = df["period"].dt.month
    df["sin_m"] = np.sin(2 * np.pi * m / 12); df["cos_m"] = np.cos(2 * np.pi * m / 12)
    df = df.dropna(subset=["lag1", "lag2"]).drop(columns="_cum")
    df = df.sort_values(["period", "state"]).reset_index(drop=True)  # chronological stream
    return df


def eval_stats(df):
    n = len(df); n_warm = int(round(WARMUP_FRAC * n))
    ev = df.iloc[n_warm:]
    return n, n_warm, len(ev), int(ev.outbreak.sum())


def build_linelist(conf, rng):
    """Case-level records: confirmed counts fixed by `conf`; other notifications added so every
    state-month has >= 1 record and the total is exactly TOTAL_RECORDS."""
    states = list(STATES); nm = len(MONTHS)
    c = conf[states].values                                         # nm x 37
    other = np.zeros_like(c)
    need = TOTAL_RECORDS - c.sum()
    w = (0.55 + 0.45 * (c + 1) / (c + 1).mean(axis=0, keepdims=True))
    w = w * np.array([SHARE[s] + 0.01 for s in states])[None, :]
    w = w / w.sum()
    other = rng.multinomial(int(need) - c.size, w.ravel()).reshape(c.shape) + 1   # every cell >= 1
    assert c.sum() + other.sum() == TOTAL_RECORDS
    recs = []
    for j, s in enumerate(states):
        nl = STATES[s]; code = "".join(ch for ch in s if ch.isalpha()).upper()[:3]
        lga_w = rng.dirichlet(np.full(nl, 0.6)); lga_w[0] = max(lga_w[0], 0.02)
        lga_w /= lga_w.sum()
        for t, p in enumerate(MONTHS):
            n_conf, n_oth = int(c[t, j]), int(other[t, j])
            classes = ["CONFIRMED"] * n_conf + list(rng.choice(["SUSPECTED", "NOT A CASE"], size=n_oth, p=[.30, .70]))
            dim = p.days_in_month
            for cls in classes:
                recs.append((p.start_time + pd.Timedelta(days=int(rng.integers(0, dim))), s,
                             f"LGA-{code}-{int(rng.choice(nl, p=lga_w)) + 1:02d}", cls))
    ll = pd.DataFrame(recs, columns=["report_date", "state", "lga", "case_classification"])
    # guarantee every one of the 774 LGAs appears at least once (one distinct record each)
    for s_, nl_ in STATES.items():
        code_ = "".join(ch for ch in s_ if ch.isalpha()).upper()[:3]
        idx = rng.permutation(ll.index[ll.state == s_].values)[:nl_]
        for i_, ix in enumerate(idx):
            ll.at[ix, "lga"] = f"LGA-{code_}-{i_ + 1:02d}"
    n = len(ll)
    isc = ll.case_classification.eq("CONFIRMED").values
    age = np.clip(rng.gamma(6.0, 5.0, n) + 1, 0, 90).round().astype(int)       # peak ~25-35 y
    sex = rng.choice(["F", "M"], size=n, p=[0.52, 0.48])
    dead = np.where(isc, rng.random(n) < 0.18, rng.random(n) < 0.03)
    outcome = np.where(dead, "DEAD", np.where(rng.random(n) < 0.04, "UNKNOWN", "ALIVE"))
    ll["age_years"] = age; ll["sex"] = sex; ll["outcome"] = outcome
    ll = ll.sort_values("report_date").reset_index(drop=True)
    ll.insert(0, "case_id", [f"SYN-{i + 1:06d}" for i in range(n)])
    ll["report_date"] = ll["report_date"].dt.normalize()
    ll["data_origin"] = "SYNTHETIC"
    return ll


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="synthetic_lassa_data"); ap.add_argument("--seeds", type=int, default=600)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    best = None
    for seed in range(1, a.seeds + 1):                    # choose a seed whose class balance matches ~5.7 %
        rng = np.random.default_rng(seed)
        conf = simulate_confirmed(rng)
        tot = int(conf.values.sum())
        if not (2700 <= tot <= 3600) or tot >= TOTAL_RECORDS - conf.size:
            continue
        sm = add_features(build_state_month(conf))
        n, nw, ne, pos = eval_stats(sm)
        score = abs(pos - TARGET_EVAL_POS)
        if best is None or score < best[0]:
            best = (score, seed, n, nw, ne, pos, tot)
        if score == 0:
            break
    _, seed, n, nw, ne, pos, tot = best
    rng = np.random.default_rng(seed); conf = simulate_confirmed(rng)
    sm = add_features(build_state_month(conf))
    ll = build_linelist(conf, np.random.default_rng(10_000 + seed))
    cols = ["state", "period", "confirmed", "lag1", "lag2", "roll3", "trend", "hotspot", "sin_m", "cos_m", "q70", "outbreak"]
    sm_out = sm[cols].copy(); sm_out["period"] = sm_out["period"].astype(str); sm_out["data_origin"] = "SYNTHETIC"
    ll.to_csv(f"{a.out}/SYNTHETIC_lassa_linelist_2018_2021.csv", index=False)
    sm_out.to_csv(f"{a.out}/SYNTHETIC_lassa_state_month_features.csv", index=False)
    try:
        import pyreadstat
        ll2 = ll.copy(); ll2["report_date"] = pd.to_datetime(ll2["report_date"])
        pyreadstat.write_sav(ll2, f"{a.out}/SYNTHETIC_lassa_linelist_2018_2021.sav",
                             column_labels=["Case ID", "Report date", "State", "LGA (synthetic code)", "Case classification",
                                            "Age (years)", "Sex", "Outcome", "Data origin"])
    except Exception as e:
        print("(.sav not written:", e, ")")
    print(f"seed={seed}  linelist rows={len(ll)}  distinct LGAs={ll.lga.nunique()}  states={ll.state.nunique()}")
    print(f"confirmed total={int(conf.values.sum())}  state-month rows={n} warm-up={nw} eval={ne} eval outbreaks={pos} "
          f"({pos / ne:.2%}); overall prevalence={sm.outbreak.mean():.2%}")


if __name__ == "__main__":
    main()

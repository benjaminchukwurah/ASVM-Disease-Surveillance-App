#!/usr/bin/env python3
"""
Load the SYNTHETIC Lassa fever data into PostgreSQL (works with Neon).

Creates schema `synthetic` (kept apart from any real data you add later) with two tables:
    synthetic.lassa_linelist     20,062 case records
    synthetic.lassa_state_month  1,702 state-month rows (features + outbreak label)

Setup (once):
    pip install "psycopg[binary]" pandas
    export DATABASE_URL='postgresql://USER:PASSWORD@HOST/DBNAME?sslmode=require'   # from Neon dashboard

Run:
    python load_to_postgres.py --data-dir .            # create tables + load (fails if data already loaded)
    python load_to_postgres.py --data-dir . --reset    # wipe the synthetic tables and reload

The connection string is read ONLY from the DATABASE_URL environment variable. Never paste it into code,
chat, or Git.
"""
import argparse, io, os, sys
import pandas as pd
import psycopg

LL_CSV = "SYNTHETIC_lassa_linelist_2018_2021.csv"
SM_CSV = "SYNTHETIC_lassa_state_month_features.csv"

DDL = """
CREATE SCHEMA IF NOT EXISTS synthetic;

CREATE TABLE IF NOT EXISTS synthetic.lassa_linelist (
    case_id             TEXT PRIMARY KEY,
    report_date         DATE        NOT NULL,
    state               TEXT        NOT NULL,
    lga                 TEXT        NOT NULL,
    case_classification TEXT        NOT NULL CHECK (case_classification IN ('CONFIRMED','SUSPECTED','NOT A CASE')),
    age_years           SMALLINT,
    sex                 CHAR(1),
    outcome             TEXT,
    data_origin         TEXT        NOT NULL DEFAULT 'SYNTHETIC'
);
CREATE INDEX IF NOT EXISTS ix_ll_date  ON synthetic.lassa_linelist (report_date);
CREATE INDEX IF NOT EXISTS ix_ll_state ON synthetic.lassa_linelist (state, report_date);

CREATE TABLE IF NOT EXISTS synthetic.lassa_state_month (
    state       TEXT             NOT NULL,
    period      DATE             NOT NULL,          -- first day of the month
    confirmed   INTEGER          NOT NULL,
    lag1        DOUBLE PRECISION NOT NULL,
    lag2        DOUBLE PRECISION NOT NULL,
    roll3       DOUBLE PRECISION NOT NULL,
    trend       DOUBLE PRECISION NOT NULL,
    hotspot     SMALLINT         NOT NULL,
    sin_m       DOUBLE PRECISION NOT NULL,
    cos_m       DOUBLE PRECISION NOT NULL,
    q70         DOUBLE PRECISION NOT NULL,
    outbreak    SMALLINT         NOT NULL CHECK (outbreak IN (0,1)),
    data_origin TEXT             NOT NULL DEFAULT 'SYNTHETIC',
    PRIMARY KEY (state, period)
);
CREATE INDEX IF NOT EXISTS ix_sm_period ON synthetic.lassa_state_month (period);
"""


def copy_df(cur, df, table, cols):
    buf = io.StringIO()
    df[cols].to_csv(buf, index=False, header=False)
    buf.seek(0)
    with cur.copy(f"COPY {table} ({', '.join(cols)}) FROM STDIN WITH (FORMAT csv)") as cp:
        while chunk := buf.read(1 << 20):
            cp.write(chunk)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=".")
    ap.add_argument("--reset", action="store_true", help="TRUNCATE the synthetic tables before loading")
    a = ap.parse_args()

    url = os.environ.get("DATABASE_URL")
    if not url:
        sys.exit("DATABASE_URL is not set. Copy the connection string from the Neon dashboard and run:\n"
                 "  export DATABASE_URL='postgresql://USER:PASSWORD@HOST/DBNAME?sslmode=require'")

    ll = pd.read_csv(os.path.join(a.data_dir, LL_CSV))
    sm = pd.read_csv(os.path.join(a.data_dir, SM_CSV))
    sm["period"] = pd.to_datetime(sm["period"] + "-01").dt.date          # 'YYYY-MM' -> first of month
    ll["sex"] = ll["sex"].astype(str)

    with psycopg.connect(url, autocommit=False) as conn:                   # one transaction: all or nothing
        with conn.cursor() as cur:
            cur.execute(DDL)
            if a.reset:
                cur.execute("TRUNCATE synthetic.lassa_linelist, synthetic.lassa_state_month")
            cur.execute("SELECT (SELECT count(*) FROM synthetic.lassa_linelist), (SELECT count(*) FROM synthetic.lassa_state_month)")
            n_ll, n_sm = cur.fetchone()
            if n_ll or n_sm:
                sys.exit(f"Tables already contain data ({n_ll} line-list rows, {n_sm} state-month rows). "
                         "Re-run with --reset to replace it.")
            copy_df(cur, ll, "synthetic.lassa_linelist",
                    ["case_id", "report_date", "state", "lga", "case_classification", "age_years", "sex", "outcome", "data_origin"])
            copy_df(cur, sm, "synthetic.lassa_state_month",
                    ["state", "period", "confirmed", "lag1", "lag2", "roll3", "trend", "hotspot", "sin_m", "cos_m", "q70", "outbreak", "data_origin"])
            cur.execute("""
                SELECT (SELECT count(*) FROM synthetic.lassa_linelist),
                       (SELECT count(DISTINCT lga)   FROM synthetic.lassa_linelist),
                       (SELECT count(DISTINCT state) FROM synthetic.lassa_linelist),
                       (SELECT count(*) FROM synthetic.lassa_state_month),
                       (SELECT sum(outbreak) FROM synthetic.lassa_state_month)""")
            r = cur.fetchone()
        conn.commit()
    print(f"Loaded: {r[0]:,} line-list rows ({r[1]} LGAs, {r[2]} states); "
          f"{r[3]:,} state-month rows with {r[4]} outbreak-labelled.")
    assert (r[0], r[1], r[2], r[3]) == (20062, 774, 37, 1702), "unexpected counts - check the input files"
    print("Row counts match the expected 20,062 / 774 / 37 / 1,702.")


if __name__ == "__main__":
    main()

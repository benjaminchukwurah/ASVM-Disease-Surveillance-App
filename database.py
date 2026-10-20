import base64
import hashlib
import hmac
import secrets
from contextlib import contextmanager
from contextlib import closing
from functools import lru_cache

import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import ThreadedConnectionPool


@lru_cache(maxsize=4)
def _connection_pool(connection_string):
    return ThreadedConnectionPool(minconn=1, maxconn=8, dsn=connection_string, connect_timeout=8)


@contextmanager
def pooled_connection(connection_string):
    pool = _connection_pool(connection_string)
    connection = pool.getconn()
    try:
        if connection.closed:
            pool.putconn(connection, close=True)
            connection = pool.getconn()
        yield connection
    finally:
        if connection.closed:
            pool.putconn(connection, close=True)
        else:
            if connection.get_transaction_status() != psycopg2.extensions.TRANSACTION_STATUS_IDLE:
                connection.rollback()
            pool.putconn(connection)


SYNTHETIC_OVERVIEW_QUERY = """
SELECT
    (SELECT COUNT(*) FROM synthetic.lassa_linelist) AS total_case_records,
    (SELECT COUNT(DISTINCT state) FROM synthetic.lassa_linelist) AS states_covered,
    (SELECT COUNT(DISTINCT lga) FROM synthetic.lassa_linelist) AS lgas_covered,
    (SELECT COUNT(*) FROM synthetic.lassa_state_month) AS state_month_rows,
    (SELECT COUNT(*) FROM synthetic.lassa_state_month WHERE outbreak = 1) AS outbreak_month_rows,
    (SELECT MIN(period) FROM synthetic.lassa_state_month) AS first_period,
    (SELECT MAX(period) FROM synthetic.lassa_state_month) AS last_period
"""


SYNTHETIC_TOP_STATES_QUERY = """
SELECT state,
       SUM(confirmed) AS total_confirmed_cases,
       SUM(CASE WHEN outbreak = 1 THEN 1 ELSE 0 END) AS outbreak_months,
       MAX(period) AS latest_period
FROM synthetic.lassa_state_month
GROUP BY state
ORDER BY total_confirmed_cases DESC
LIMIT %s
"""

SYNTHETIC_MONTHLY_SUMMARY_QUERY = """
SELECT period,
       SUM(confirmed) AS total_confirmed,
       SUM(outbreak) AS outbreak_months,
       COUNT(*) AS state_count
FROM synthetic.lassa_state_month
GROUP BY period
ORDER BY period ASC
"""

SYNTHETIC_STATE_LIST_QUERY = """
SELECT DISTINCT state
FROM synthetic.lassa_state_month
ORDER BY state ASC
"""

SYNTHETIC_STATE_HISTORY_QUERY = """
SELECT state, period, confirmed, outbreak, q70, lag1, lag2, roll3, trend, hotspot
FROM synthetic.lassa_state_month
WHERE state = %s
ORDER BY period ASC
"""

SYNTHETIC_MONITOR_QUERY = """
WITH latest AS (
    SELECT DISTINCT ON (state)
        state,
        period,
        confirmed,
        q70,
        outbreak,
        lag1,
        lag2,
        trend,
        hotspot
    FROM synthetic.lassa_state_month
    ORDER BY state, period DESC
)
SELECT *
FROM latest
ORDER BY outbreak DESC, confirmed DESC, state ASC
"""

SYNTHETIC_ALERT_QUERY = """
WITH latest AS (
    SELECT DISTINCT ON (state)
        state,
        period,
        confirmed,
        outbreak,
        q70,
        hotspot
    FROM synthetic.lassa_state_month
    ORDER BY state, period DESC
)
SELECT state, period, confirmed, q70, outbreak,
       CASE
           WHEN outbreak = 1 THEN 'HIGH'
           WHEN confirmed >= q70 * 0.8 THEN 'MEDIUM'
           ELSE 'LOW'
       END AS risk_label
FROM latest
WHERE outbreak = 1 OR confirmed >= q70 * 0.8
ORDER BY confirmed DESC, state ASC
LIMIT %s
"""


CREATE_CASES_TABLE = """
CREATE TABLE IF NOT EXISTS surveillance_cases (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    disease VARCHAR(120) NOT NULL,
    report_date DATE NOT NULL,
    state VARCHAR(100) NOT NULL,
    lga VARCHAR(120) NOT NULL,
    age_group VARCHAR(40) NOT NULL,
    sex VARCHAR(30) NOT NULL,
    symptoms TEXT[] NOT NULL DEFAULT '{}',
    other_symptoms TEXT NOT NULL DEFAULT '',
    environmental_exposure TEXT NOT NULL DEFAULT '',
    clinical_notes TEXT NOT NULL DEFAULT '',
    laboratory_status VARCHAR(40) NOT NULL DEFAULT 'Pending',
    laboratory_findings TEXT NOT NULL DEFAULT '',
    supporting_document_name VARCHAR(255) NOT NULL DEFAULT '',
    case_status VARCHAR(40) NOT NULL DEFAULT 'Suspected',
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
)
"""

CREATE_USERS_TABLE = """
CREATE TABLE IF NOT EXISTS surveillance_users (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    username VARCHAR(80) NOT NULL,
    password_hash TEXT NOT NULL,
    role VARCHAR(40) NOT NULL CHECK (role IN ('Health Worker', 'Supervisor')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE UNIQUE INDEX IF NOT EXISTS ix_surveillance_users_username
    ON surveillance_users (LOWER(username));
"""

PASSWORD_HASH_ROUNDS = 310_000


def hash_user_password(password):
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PASSWORD_HASH_ROUNDS)
    return "$".join((
        "pbkdf2_sha256",
        str(PASSWORD_HASH_ROUNDS),
        base64.urlsafe_b64encode(salt).decode("ascii"),
        base64.urlsafe_b64encode(digest).decode("ascii"),
    ))


def verify_user_password(password, encoded_hash):
    try:
        algorithm, rounds, encoded_salt, encoded_digest = encoded_hash.split("$", 3)
        if algorithm != "pbkdf2_sha256" or int(rounds) != PASSWORD_HASH_ROUNDS:
            return False
        salt = base64.urlsafe_b64decode(encoded_salt)
        expected = base64.urlsafe_b64decode(encoded_digest)
    except (AttributeError, ValueError):
        return False
    actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PASSWORD_HASH_ROUNDS)
    return hmac.compare_digest(actual, expected)


def ensure_user_schema(connection_string):
    with closing(psycopg2.connect(connection_string, connect_timeout=8)) as connection:
        with connection:
            with connection.cursor() as cursor:
                cursor.execute(CREATE_USERS_TABLE)


def authenticate_database_user(connection_string, username, password):
    ensure_user_schema(connection_string)
    with closing(psycopg2.connect(connection_string, connect_timeout=8)) as connection:
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                "SELECT username, password_hash, role FROM surveillance_users WHERE LOWER(username) = LOWER(%s)",
                (username.strip(),),
            )
            account = cursor.fetchone()
    if account and verify_user_password(password, account["password_hash"]):
        return {"username": account["username"], "role": account["role"]}
    return None


def create_user_account(connection_string, username, password, role):
    username = username.strip()
    if not 3 <= len(username) <= 80:
        raise ValueError("Username must be between 3 and 80 characters.")
    if len(password) < 12:
        raise ValueError("Password must be at least 12 characters.")
    if role not in ("Health Worker", "Supervisor"):
        raise ValueError("Select a valid role.")

    ensure_user_schema(connection_string)
    try:
        with closing(psycopg2.connect(connection_string, connect_timeout=8)) as connection:
            with connection:
                with connection.cursor() as cursor:
                    cursor.execute(
                        "INSERT INTO surveillance_users (username, password_hash, role) VALUES (%s, %s, %s)",
                        (username, hash_user_password(password), role),
                    )
    except psycopg2.errors.UniqueViolation:
        raise ValueError("That username is already in use.") from None


def get_user_accounts(connection_string):
    ensure_user_schema(connection_string)
    with closing(psycopg2.connect(connection_string, connect_timeout=8)) as connection:
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("SELECT username, role, created_at FROM surveillance_users ORDER BY username")
            return [dict(row) for row in cursor.fetchall()]


INSERT_CASE = """
INSERT INTO surveillance_cases (
    disease, report_date, state, lga, age_group, sex, symptoms,
    other_symptoms, environmental_exposure, clinical_notes,
    laboratory_status, laboratory_findings, supporting_document_name
) VALUES (
    %(disease)s, %(report_date)s, %(state)s, %(lga)s, %(age_group)s,
    %(sex)s, %(symptoms)s, %(other_symptoms)s, %(environmental_exposure)s,
    %(clinical_notes)s, %(laboratory_status)s, %(laboratory_findings)s,
    %(supporting_document_name)s
)
RETURNING id, created_at
"""


def initialize_database(connection_string):
    with closing(psycopg2.connect(connection_string, connect_timeout=8)) as connection:
        with connection:
            with connection.cursor() as cursor:
                cursor.execute(CREATE_CASES_TABLE)


def save_case(connection_string, case):
    with closing(psycopg2.connect(connection_string, connect_timeout=8)) as connection:
        with connection:
            with connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(INSERT_CASE, case)
                return dict(cursor.fetchone())


def get_recent_cases(connection_string, limit=25):
    query = """
    SELECT id, disease, report_date, state, lga, age_group, sex,
           laboratory_status, case_status, created_at
    FROM surveillance_cases
    ORDER BY created_at DESC
    LIMIT %s
    """
    with pooled_connection(connection_string) as connection:
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query, (limit,))
            return [dict(row) for row in cursor.fetchall()]


def get_overview_data(connection_string, limit=8):
    with pooled_connection(connection_string) as connection:
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(SYNTHETIC_OVERVIEW_QUERY)
            overview = cursor.fetchone()
            cursor.execute(SYNTHETIC_TOP_STATES_QUERY, (limit,))
            top_states = [dict(row) for row in cursor.fetchall()]
    return dict(overview) if overview else {}, top_states


def get_trends_data(connection_string):
    with pooled_connection(connection_string) as connection:
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(SYNTHETIC_MONTHLY_SUMMARY_QUERY)
            summary = [dict(row) for row in cursor.fetchall()]
            cursor.execute(SYNTHETIC_STATE_LIST_QUERY)
            states = [row["state"] for row in cursor.fetchall()]
    return summary, states


def get_synthetic_overview(connection_string):
    with pooled_connection(connection_string) as connection:
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(SYNTHETIC_OVERVIEW_QUERY)
            row = cursor.fetchone()
            return dict(row) if row else {}


def get_synthetic_top_states(connection_string, limit=8):
    with pooled_connection(connection_string) as connection:
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(SYNTHETIC_TOP_STATES_QUERY, (limit,))
            return [dict(row) for row in cursor.fetchall()]


def get_synthetic_monthly_summary(connection_string):
    with pooled_connection(connection_string) as connection:
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(SYNTHETIC_MONTHLY_SUMMARY_QUERY)
            return [dict(row) for row in cursor.fetchall()]


def get_synthetic_states(connection_string):
    with pooled_connection(connection_string) as connection:
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(SYNTHETIC_STATE_LIST_QUERY)
            return [row["state"] for row in cursor.fetchall()]


def get_synthetic_state_history(connection_string, state):
    with pooled_connection(connection_string) as connection:
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(SYNTHETIC_STATE_HISTORY_QUERY, (state,))
            return [dict(row) for row in cursor.fetchall()]


def get_synthetic_monitor_rows(connection_string):
    with pooled_connection(connection_string) as connection:
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(SYNTHETIC_MONITOR_QUERY)
            return [dict(row) for row in cursor.fetchall()]


def get_synthetic_alerts(connection_string, limit=10):
    with pooled_connection(connection_string) as connection:
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(SYNTHETIC_ALERT_QUERY, (limit,))
            return [dict(row) for row in cursor.fetchall()]
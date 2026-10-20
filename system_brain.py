from datetime import datetime, timedelta, timezone

import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler

from asvm_model import AdaptiveSVM, FEATURE_COLUMNS, MODEL_VERSION, classification_metrics
from database import pooled_connection

BATCH_SIZE = 100
WARMUP_FRACTION = 0.15
MAX_ALERTS_PER_CYCLE = 8
ALERT_COOLDOWN = timedelta(hours=24)
DISEASE = "Lassa fever"

CREATE_BRAIN_TABLES = """
CREATE TABLE IF NOT EXISTS surveillance_brain_runs (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    started_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    finished_at TIMESTAMPTZ,
    status VARCHAR(30) NOT NULL,
    records_processed INTEGER NOT NULL DEFAULT 0,
    warmup_records INTEGER NOT NULL DEFAULT 0,
    predictions_made INTEGER NOT NULL DEFAULT 0,
    alerts_generated INTEGER NOT NULL DEFAULT 0,
    message TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS surveillance_brain_model (
    id SMALLINT PRIMARY KEY CHECK (id = 1),
    checkpoint BYTEA NOT NULL,
    next_index INTEGER NOT NULL,
    model_version VARCHAR(40) NOT NULL DEFAULT 'legacy-v1',
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS surveillance_brain_predictions (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    run_id BIGINT NOT NULL REFERENCES surveillance_brain_runs(id),
    state VARCHAR(100) NOT NULL,
    period DATE NOT NULL,
    decision_score DOUBLE PRECISION NOT NULL,
    probability DOUBLE PRECISION,
    risk_category VARCHAR(20) NOT NULL,
    actual_outbreak SMALLINT NOT NULL,
    model_version VARCHAR(40) NOT NULL DEFAULT 'legacy-v1',
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (state, period)
);
CREATE TABLE IF NOT EXISTS surveillance_brain_alerts (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    state VARCHAR(100) NOT NULL,
    disease VARCHAR(120) NOT NULL,
    period DATE NOT NULL,
    probability DOUBLE PRECISION NOT NULL,
    risk_category VARCHAR(10) NOT NULL,
    run_id BIGINT NOT NULL REFERENCES surveillance_brain_runs(id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS surveillance_brain_baseline (
    id SMALLINT PRIMARY KEY CHECK (id = 1),
    model_version VARCHAR(40) NOT NULL,
    train_rows INTEGER NOT NULL,
    test_rows INTEGER NOT NULL,
    test_start DATE NOT NULL,
    true_negative INTEGER NOT NULL,
    false_positive INTEGER NOT NULL,
    false_negative INTEGER NOT NULL,
    true_positive INTEGER NOT NULL,
    precision DOUBLE PRECISION NOT NULL,
    recall DOUBLE PRECISION NOT NULL,
    f1 DOUBLE PRECISION NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS ix_brain_alert_cooldown
    ON surveillance_brain_alerts (state, disease, created_at DESC);
CREATE INDEX IF NOT EXISTS ix_brain_prediction_period
    ON surveillance_brain_predictions (period DESC, risk_category);
ALTER TABLE surveillance_brain_predictions
    ALTER COLUMN risk_category TYPE VARCHAR(20);
ALTER TABLE surveillance_brain_alerts
    ALTER COLUMN risk_category TYPE VARCHAR(20);
ALTER TABLE surveillance_brain_model
    ADD COLUMN IF NOT EXISTS model_version VARCHAR(40) NOT NULL DEFAULT 'legacy-v1';
ALTER TABLE surveillance_brain_predictions
    ADD COLUMN IF NOT EXISTS model_version VARCHAR(40) NOT NULL DEFAULT 'legacy-v1';
"""

SOURCE_QUERY = """
SELECT state, period, confirmed, lag1, lag2, roll3, trend, hotspot, sin_m, cos_m
FROM synthetic.lassa_state_month
WHERE data_origin = 'SYNTHETIC'
ORDER BY period ASC, state ASC
"""

HISTORICAL_CONFIRMED_QUERY = """
SELECT state, date_trunc('month', report_date)::date AS period, COUNT(*) AS confirmed
FROM synthetic.lassa_linelist
WHERE case_classification = 'CONFIRMED'
  AND data_origin = 'SYNTHETIC'
GROUP BY state, date_trunc('month', report_date)::date
ORDER BY state, period
"""

INSERT_PREDICTIONS = """
INSERT INTO surveillance_brain_predictions (
    run_id, state, period, decision_score, probability, risk_category, actual_outbreak, model_version
) VALUES %s
ON CONFLICT (state, period) DO UPDATE SET
    run_id = EXCLUDED.run_id,
    decision_score = EXCLUDED.decision_score,
    probability = EXCLUDED.probability,
    risk_category = EXCLUDED.risk_category,
    actual_outbreak = EXCLUDED.actual_outbreak,
    model_version = EXCLUDED.model_version,
    created_at = CURRENT_TIMESTAMP
"""


def evaluate_linear_svc_baseline(observations):
    """Fit a single LinearSVC on the first 70% of months and test on the last 30%."""
    periods = sorted({row["period"] for row in observations})
    if len(periods) < 2:
        raise ValueError("At least two distinct periods are required for a temporal baseline split.")
    split_index = min(max(int(round(0.7 * len(periods))), 1), len(periods) - 1)
    test_start = periods[split_index]
    train = [row for row in observations if row["period"] < test_start]
    test = [row for row in observations if row["period"] >= test_start]
    train_labels = np.asarray([row["target"] for row in train], dtype=int)
    test_labels = np.asarray([row["target"] for row in test], dtype=int)
    if len(np.unique(train_labels)) != 2:
        raise ValueError("The LinearSVC training period must contain both outbreak classes.")

    scaler = StandardScaler().fit([row["features"] for row in train])
    train_features = scaler.transform([row["features"] for row in train])
    test_features = scaler.transform([row["features"] for row in test])
    baseline = LinearSVC(class_weight="balanced", random_state=42, max_iter=10000)
    baseline.fit(train_features, train_labels)
    metrics = classification_metrics(baseline.decision_function(test_features), test_labels)
    return {
        "train_rows": len(train),
        "test_rows": len(test),
        "test_start": test_start,
        **metrics,
    }


def _save_linear_svc_baseline(connection, observations):
    result = evaluate_linear_svc_baseline(observations)
    with connection.cursor() as cursor:
        cursor.execute(
            """INSERT INTO surveillance_brain_baseline (
                   id, model_version, train_rows, test_rows, test_start,
                   true_negative, false_positive, false_negative, true_positive,
                   precision, recall, f1
               ) VALUES (1, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
               ON CONFLICT (id) DO UPDATE SET
                   model_version = EXCLUDED.model_version,
                   train_rows = EXCLUDED.train_rows,
                   test_rows = EXCLUDED.test_rows,
                   test_start = EXCLUDED.test_start,
                   true_negative = EXCLUDED.true_negative,
                   false_positive = EXCLUDED.false_positive,
                   false_negative = EXCLUDED.false_negative,
                   true_positive = EXCLUDED.true_positive,
                   precision = EXCLUDED.precision,
                   recall = EXCLUDED.recall,
                   f1 = EXCLUDED.f1,
                   created_at = CURRENT_TIMESTAMP""",
            (
                MODEL_VERSION, result["train_rows"], result["test_rows"], result["test_start"],
                result["true_negative"], result["false_positive"], result["false_negative"],
                result["true_positive"], result["precision"], result["recall"], result["f1"],
            ),
        )
    return result


def _load_observations(connection):
    with connection.cursor(cursor_factory=RealDictCursor) as cursor:
        cursor.execute(SOURCE_QUERY)
        observations = [dict(row) for row in cursor.fetchall()]
        cursor.execute(HISTORICAL_CONFIRMED_QUERY)
        confirmed_history = {}
        for row in cursor.fetchall():
            confirmed_history.setdefault(row["state"], {})[row["period"]] = int(row["confirmed"])

    prior_counts = {state: [] for state in confirmed_history}
    observations_by_state = {}
    for row in observations:
        observations_by_state.setdefault(row["state"], []).append(row)

    for state, rows in observations_by_state.items():
        rows.sort(key=lambda item: item["period"])
        monthly_confirmed = confirmed_history.get(state, {})
        if rows:
            first_period = rows[0]["period"]
            history_start = first_period.replace(
                year=first_period.year - (1 if first_period.month <= 2 else 0),
                month=((first_period.month - 3) % 12) + 1,
            )
            for month_offset in range(2):
                month_index = history_start.month - 1 + month_offset
                year = history_start.year + month_index // 12
                month = month_index % 12 + 1
                period = first_period.replace(year=year, month=month)
                prior_counts.setdefault(state, []).append(monthly_confirmed.get(period, 0))

    for row in observations:
        history = prior_counts.setdefault(row["state"], [])
        threshold = float(np.percentile(history, 70)) if history else None
        row["target"] = int(
            threshold is not None
            and int(row["confirmed"]) > threshold
            and int(row["confirmed"]) >= 3
        )
        history.append(int(row["confirmed"]))
        row["features"] = [float(row[column]) for column in FEATURE_COLUMNS]
    observations.sort(key=lambda item: (item["period"], item["state"]))
    return observations


def _eligible_alerts(connection, latest_rows, run_id):
    now = datetime.now(timezone.utc)
    eligible = sorted(
        (row for row in latest_rows if row["risk_category"] == "HIGH" and row["probability"] is not None),
        key=lambda row: (-row["probability"], row["state"]),
    )
    created = 0
    with connection.cursor() as cursor:
        for row in eligible:
            if created >= MAX_ALERTS_PER_CYCLE:
                break
            cursor.execute(
                """SELECT 1 FROM surveillance_brain_alerts
                   WHERE state = %s AND disease = %s AND created_at >= %s LIMIT 1""",
                (row["state"], DISEASE, now - ALERT_COOLDOWN),
            )
            if cursor.fetchone():
                continue
            cursor.execute(
                """INSERT INTO surveillance_brain_alerts
                   (state, disease, period, probability, risk_category, run_id)
                   VALUES (%s, %s, %s, %s, 'HIGH', %s)""",
                (row["state"], DISEASE, row["period"], row["probability"], run_id),
            )
            created += 1
    return created


def run_cycle(connection_string):
    """Run one persisted Sense -> Think -> Act -> Learn cycle."""
    with pooled_connection(connection_string) as connection:
        with connection.cursor() as cursor:
            cursor.execute(CREATE_BRAIN_TABLES)
        connection.commit()

        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                "INSERT INTO surveillance_brain_runs (status, message) VALUES ('RUNNING', 'Cycle started') RETURNING id"
            )
            run_id = cursor.fetchone()["id"]
        connection.commit()

        try:
            with connection:
                with connection.cursor() as cursor:
                    cursor.execute("SELECT pg_advisory_xact_lock(hashtext('asvm_system_brain_cycle'))")
                observations = _load_observations(connection)
                if not observations:
                    raise ValueError("No synthetic state-month observations are available.")

                with connection.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute("SELECT model_version FROM surveillance_brain_baseline WHERE id = 1")
                    baseline_row = cursor.fetchone()
                if baseline_row is None or baseline_row["model_version"] != MODEL_VERSION:
                    _save_linear_svc_baseline(connection, observations)

                with connection.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute("SELECT checkpoint, next_index, model_version FROM surveillance_brain_model WHERE id = 1 FOR UPDATE")
                    saved = cursor.fetchone()
                if saved and saved["model_version"] != MODEL_VERSION:
                    saved = None

                model = AdaptiveSVM.from_bytes(saved["checkpoint"]) if saved else AdaptiveSVM()
                next_index = int(saved["next_index"]) if saved else 0
                if next_index > len(observations):
                    raise ValueError("Stored model cursor is beyond the current source data; reset the model checkpoint before continuing.")

                warmup_records = 0
                processed = 0
                predictions = 0
                alert_rows = []
                if saved is None:
                    warmup_end = max(1, int(round(WARMUP_FRACTION * len(observations))))
                    warmup = observations[:warmup_end]
                    model.warm_up(
                        [row["features"] for row in warmup],
                        [row["target"] for row in warmup],
                    )
                    next_index = warmup_end
                    warmup_records = len(warmup)
                    processed += warmup_records

                latest_period = observations[-1]["period"]
                while next_index < len(observations):
                    batch = observations[next_index:next_index + BATCH_SIZE]
                    features = [row["features"] for row in batch]
                    labels = np.asarray([row["target"] for row in batch], dtype=int)
                    scores = model.decision_function(features)
                    probabilities = model.predict_proba(features)[:, 1]

                    batch_predictions = []
                    for row, score, probability in zip(batch, scores, probabilities):
                        category = AdaptiveSVM.get_risk_category(probability)
                        prediction = {
                            "state": row["state"],
                            "period": row["period"],
                            "decision_score": float(score),
                            "probability": float(probability) if probability is not None else None,
                            "risk_category": category,
                            "actual_outbreak": row["target"],
                        }
                        batch_predictions.append(prediction)
                        if row["period"] == latest_period:
                            alert_rows.append(prediction)

                    if batch_predictions:
                        with connection.cursor() as cursor:
                            execute_values(
                                cursor,
                                INSERT_PREDICTIONS,
                                [(
                                    run_id, row["state"], row["period"], row["decision_score"],
                                    row["probability"], row["risk_category"], row["actual_outbreak"], MODEL_VERSION,
                                ) for row in batch_predictions],
                            )
                    predictions += len(batch_predictions)
                    processed += len(batch)

                    model.partial_fit(features, labels)
                    next_index += len(batch)

                alert_count = _eligible_alerts(connection, alert_rows, run_id)
                checkpoint = model.to_bytes()
                with connection.cursor() as cursor:
                    cursor.execute(
                        """INSERT INTO surveillance_brain_model (id, checkpoint, next_index, model_version)
                           VALUES (1, %s, %s, %s)
                           ON CONFLICT (id) DO UPDATE SET checkpoint = EXCLUDED.checkpoint,
                               next_index = EXCLUDED.next_index, model_version = EXCLUDED.model_version,
                               updated_at = CURRENT_TIMESTAMP""",
                        (psycopg2.Binary(checkpoint), next_index, MODEL_VERSION),
                    )
                    cursor.execute(
                        """UPDATE surveillance_brain_runs
                           SET finished_at = CURRENT_TIMESTAMP, status = %s, records_processed = %s,
                               warmup_records = %s, predictions_made = %s, alerts_generated = %s,
                               message = %s WHERE id = %s""",
                        (
                            "COMPLETED" if predictions else "NO_NEW_DATA",
                            processed,
                            warmup_records,
                            predictions,
                            alert_count,
                            "Cycle completed" if predictions else "No new state-month observations",
                            run_id,
                        ),
                    )
            return get_run(run_id, connection_string)
        except Exception as error:
            connection.rollback()
            with connection:
                with connection.cursor() as cursor:
                    cursor.execute(
                        """UPDATE surveillance_brain_runs
                           SET finished_at = CURRENT_TIMESTAMP, status = 'FAILED', message = %s
                           WHERE id = %s""",
                        ("Cycle failed; check database connectivity and application logs.", run_id),
                    )
            raise


def get_run(run_id, connection_string):
    with pooled_connection(connection_string) as connection:
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("SELECT * FROM surveillance_brain_runs WHERE id = %s", (run_id,))
            row = cursor.fetchone()
            return dict(row) if row else None


def get_brain_status(connection_string):
    with pooled_connection(connection_string) as connection:
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("SELECT * FROM surveillance_brain_runs ORDER BY id DESC LIMIT 1")
            run = cursor.fetchone()
            cursor.execute("SELECT next_index, model_version, updated_at FROM surveillance_brain_model WHERE id = 1")
            model = cursor.fetchone()
            cursor.execute(
                """SELECT state, disease, period, probability, created_at
                   FROM surveillance_brain_alerts ORDER BY created_at DESC LIMIT 100"""
            )
            alerts = [dict(row) for row in cursor.fetchall()]
            cursor.execute(
                """SELECT state, period, decision_score, probability, risk_category, actual_outbreak
                   FROM surveillance_brain_predictions
                   WHERE model_version = %s
                   ORDER BY period DESC, state ASC LIMIT 100""",
                (MODEL_VERSION,),
            )
            predictions = [dict(row) for row in cursor.fetchall()]
            cursor.execute(
                """SELECT
                       COUNT(*) AS evaluated,
                       COUNT(*) FILTER (WHERE decision_score > 0 AND actual_outbreak = 1) AS true_positive,
                       COUNT(*) FILTER (WHERE decision_score > 0 AND actual_outbreak = 0) AS false_positive,
                       COUNT(*) FILTER (WHERE decision_score <= 0 AND actual_outbreak = 1) AS false_negative,
                       COUNT(*) FILTER (WHERE decision_score <= 0 AND actual_outbreak = 0) AS true_negative
                   FROM surveillance_brain_predictions
                   WHERE model_version = %s""",
                (MODEL_VERSION,),
            )
            evaluation = dict(cursor.fetchone())
            cursor.execute("SELECT * FROM surveillance_brain_baseline WHERE id = 1 AND model_version = %s", (MODEL_VERSION,))
            baseline = cursor.fetchone()
            cursor.execute("SELECT * FROM surveillance_brain_runs ORDER BY id DESC LIMIT 20")
            runs = [dict(row) for row in cursor.fetchall()]
    return {
        "last_run": dict(run) if run else None,
        "model": dict(model) if model else None,
        "model_ready": bool(model and model["model_version"] == MODEL_VERSION),
        "alerts": alerts,
        "predictions": predictions,
        "evaluation": evaluation,
        "baseline": dict(baseline) if baseline else None,
        "runs": runs,
    }

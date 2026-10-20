import hmac
import os
from datetime import date

import streamlit as st

from asvm_model import MODEL_VERSION
from database import (
    get_recent_cases,
    get_overview_data,
    get_synthetic_state_history,
    get_synthetic_monitor_rows,
    get_synthetic_alerts,
    get_trends_data,
    initialize_database,
    save_case,
    authenticate_database_user,
    create_user_account,
    get_user_accounts,
)
from system_brain import get_brain_status, run_cycle


st.set_page_config(
    page_title="Disease Surveillance",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="auto",
)

PAGES = {
    "Overview": "A clear starting point for the surveillance workspace.",
    "Register a Case": "Record a suspected case for review.",
    "Outbreak Monitor": "Review state-level surveillance risk.",
    "Trends": "Explore disease activity over time.",
    "Alerts": "Review and manage surveillance alerts.",
    "System Brain": "Inspect the surveillance engine and its activity.",
    "Model": "Review model approach, evaluation, and coverage.",
    "User Management": "Create accounts and assign workspace roles.",
}

ROLE_PAGES = {
    "Health Worker": ("Register a Case", "Overview", "Outbreak Monitor", "Trends"),
    "Supervisor": tuple(PAGES),
}

LANDING_PAGE = {
    "Health Worker": "Register a Case",
    "Supervisor": "System Brain",
}

DISEASES = (
    "Lassa fever",
    "Cholera",
    "Measles",
    "Yellow fever",
    "COVID-19",
    "Other / not listed",
)

NIGERIAN_STATES = (
    "Abia", "Adamawa", "Akwa Ibom", "Anambra", "Bauchi", "Bayelsa",
    "Benue", "Borno", "Cross River", "Delta", "Ebonyi", "Edo", "Ekiti",
    "Enugu", "FCT Abuja", "Gombe", "Imo", "Jigawa", "Kaduna", "Kano",
    "Katsina", "Kebbi", "Kogi", "Kwara", "Lagos", "Nasarawa", "Niger",
    "Ogun", "Ondo", "Osun", "Oyo", "Plateau", "Rivers", "Sokoto",
    "Taraba", "Yobe", "Zamfara",
)

COMMON_SYMPTOMS = (
    "Fever", "Headache", "Weakness / fatigue", "Muscle pain", "Joint pain",
    "Sore throat", "Cough", "Difficulty breathing", "Chest pain", "Nausea",
    "Vomiting", "Diarrhoea", "Abdominal pain", "Loss of appetite",
    "Bleeding", "Rash", "Jaundice", "Swollen face", "Hearing difficulty",
    "Confusion", "Seizures", "Other symptom (describe below)",
)


def load_demo_users():
    try:
        return st.secrets["demo_users"]
    except (KeyError, FileNotFoundError):
        return {}


def authenticate(username, password):
    normalized_username = username.strip().casefold()
    for account in DEMO_USERS.values():
        configured_username = str(account.get("username", ""))
        configured_password = str(account.get("password", ""))
        if (
            normalized_username == configured_username.casefold()
            and hmac.compare_digest(password, configured_password)
        ):
            role = str(account.get("role", ""))
            if role in ROLE_PAGES:
                return {"username": configured_username, "role": role}
    connection_string = get_database_connection_string()
    if connection_string:
        return authenticate_database_user(connection_string, username, password)
    return None


def add_styles():
    st.markdown(
        """
        <style>
        :root {
            --ink: #18332c;
            --muted: #53675f;
            --green: #24745e;
            --green-dark: #195c4a;
            --mint: #e8f2ec;
            --line: #dce7e1;
            --paper: #ffffff;
            --canvas: #f4f7f5;
        }
        [data-testid="stAppViewContainer"] { background: var(--canvas); }
        [data-testid="stHeader"] { background: var(--canvas); }
        [data-testid="stHeader"] button,
        [data-testid="stSidebarCollapsedControl"] button,
        [data-testid="stSidebarCollapseButton"] button {
            color: var(--ink) !important;
            opacity: 1 !important;
        }
        [data-testid="stHeader"] button:hover,
        [data-testid="stSidebarCollapsedControl"] button:hover,
        [data-testid="stSidebarCollapseButton"] button:hover {
            background: #e3ece6 !important;
        }
        [data-testid="stHeader"] svg,
        [data-testid="stSidebarCollapsedControl"] svg,
        [data-testid="stSidebarCollapseButton"] svg {
            color: var(--ink) !important;
            fill: var(--ink) !important;
        }
        [data-testid="stSidebar"] {
            background: var(--mint);
            border-right: 1px solid var(--line);
        }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
            color: var(--ink);
        }
        h1, h2, h3 { color: var(--ink); }
        h1 { font-size: 2rem; font-weight: 680; }
        h2, h3 { font-weight: 630; }
        p, label, [data-testid="stCaptionContainer"] {
            color: var(--muted);
        }
        [data-testid="stWidgetLabel"] p,
        [data-testid="stWidgetLabel"] label,
        [data-testid="stMarkdownContainer"] strong {
            color: var(--ink) !important;
        }
        input, textarea,
        [data-baseweb="select"] > div,
        [data-baseweb="input"] > div {
            color: var(--ink) !important;
            background-color: var(--paper) !important;
            border-color: #bdcec4 !important;
        }
        input::placeholder, textarea::placeholder { color: #687c73 !important; }
        [data-testid="stAlert"] p,
        [data-testid="stAlert"] li {
            color: var(--ink) !important;
        }
        [data-testid="stMetricLabel"], [data-testid="stMetricValue"],
        [data-testid="stMetricDelta"] { color: var(--ink) !important; }
        [data-testid="stRadio"] label {
            padding: 0.34rem 0.45rem;
            border-radius: 6px;
        }
        [data-testid="stRadio"] label:hover { background: #dbeae1; }
        [data-testid="stForm"] {
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 1.25rem;
            background: var(--paper);
        }
        [data-testid="stFormSubmitButton"] button,
        [data-testid="stFormSubmitButton"] button[kind="primary"],
        .stButton > button[kind="primary"] {
            background: var(--green-dark) !important;
            border: 1px solid var(--green-dark) !important;
            color: #ffffff !important;
            font-weight: 650;
            min-height: 2.75rem;
            transition: background-color 120ms ease, border-color 120ms ease;
        }
        [data-testid="stFormSubmitButton"] button:hover,
        .stButton > button[kind="primary"]:hover {
            background: #124b3d !important;
            border-color: #124b3d !important;
            color: #ffffff !important;
        }
        [data-testid="stFormSubmitButton"] button p,
        .stButton > button[kind="primary"] p {
            color: #ffffff !important;
        }
        .stButton > button[kind="secondary"] {
            color: var(--ink) !important;
            background: #ffffff !important;
            border-color: #bdcec4 !important;
        }
        [data-testid="stAlert"] { border-radius: 6px; }
        .brand-lockup {
            display: flex;
            align-items: center;
            gap: 0.65rem;
            margin: 0.25rem 0 1.3rem;
        }
        .brand-mark {
            display: grid;
            place-items: center;
            width: 2.25rem;
            height: 2.25rem;
            border-radius: 7px;
            background: var(--green);
            color: white;
            font-size: 1.3rem;
            font-weight: 700;
        }
        .brand-name { color: var(--ink); font-size: 1.05rem; font-weight: 700; }
        .brand-subtitle { color: var(--muted); font-size: 0.76rem; }
        .eyebrow {
            color: var(--green);
            font-size: 0.76rem;
            font-weight: 700;
            margin-bottom: 0.3rem;
        }
        .role-chip {
            display: inline-block;
            padding: 0.2rem 0.55rem;
            border: 1px solid #c6ddd1;
            border-radius: 999px;
            color: var(--green-dark);
            background: #f4faf6;
            font-size: 0.76rem;
            font-weight: 650;
        }
        [data-testid="stSidebar"] .stButton > button {
            display: flex;
            align-items: center;
            justify-content: flex-start;
            width: 100%;
            padding: 0.72rem 0.8rem;
            border-radius: 6px;
            border: 1px solid transparent;
            background: transparent;
            color: var(--ink);
            font-weight: 600;
            text-align: left;
            margin: 0.08rem 0;
        }
        [data-testid="stSidebar"] .stButton > button:hover {
            background: rgba(255,255,255,0.45);
            border-color: transparent;
        }
        [data-testid="stSidebar"] .stButton > button[kind="primary"] {
            background: rgba(255,255,255,0.65);
            border-color: transparent;
            color: var(--green-dark);
            box-shadow: inset 3px 0 0 var(--green);
        }
        .empty-state {
            margin-top: 1.35rem;
            padding: 1.1rem 1.2rem;
            border: 1px solid var(--line);
            border-left: 4px solid var(--green);
            border-radius: 6px;
            background: var(--paper);
            color: var(--muted);
        }
        .empty-state strong { color: var(--ink); }
        @media (max-width: 640px) {
            [data-testid="stAppViewContainer"] .block-container {
                padding: 3.5rem 1rem 2rem;
            }
            h1 { font-size: 1.65rem; }
            [data-testid="stForm"] {
                padding: 0.9rem;
                border-radius: 7px;
            }
            [data-testid="stHorizontalBlock"]:has([data-testid="stForm"]) > [data-testid="column"]:not(:has([data-testid="stForm"])) {
                display: none !important;
            }
            [data-testid="stHorizontalBlock"]:has([data-testid="stForm"]) > [data-testid="column"]:has([data-testid="stForm"]) {
                flex: 1 1 100% !important;
                width: 100% !important;
                min-width: 100% !important;
            }
            [data-testid="stSidebar"] {
                min-width: min(84vw, 20rem);
            }
            [data-testid="stExpandSidebarButton"] {
                display: grid !important;
                place-items: center !important;
                width: 2.75rem !important;
                height: 2.75rem !important;
                border: 1px solid var(--line) !important;
                border-radius: 7px !important;
                background: #ffffff !important;
                color: var(--ink) !important;
                box-shadow: 0 1px 3px rgba(24, 51, 44, 0.12);
            }
            [data-testid="stExpandSidebarButton"] [data-testid="stIconMaterial"] {
                font-size: 0 !important;
            }
            [data-testid="stExpandSidebarButton"]::after {
                content: "☰";
                color: var(--ink);
                font-size: 1.4rem;
                line-height: 1;
            }
            [data-testid="stHeader"] {
                background: rgba(244, 247, 245, 0.98);
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_login():
    left, center, right = st.columns([1, 1.15, 1])
    with center:
        st.markdown(
            """
            <div class="brand-lockup">
                <div class="brand-mark">+</div>
                <div>
                    <div class="brand-name">Disease Surveillance</div>
                    <div class="brand-subtitle">Public health monitoring workspace</div>
                </div>
            </div>
            <div class="eyebrow">SECURE WORKSPACE</div>
            """,
            unsafe_allow_html=True,
        )
        st.title("Sign in")

        with st.form("login_form"):
            username = st.text_input("Username", autocomplete="username")
            password = st.text_input(
                "Password",
                type="password",
                autocomplete="current-password",
            )
            submitted = st.form_submit_button(
                "Sign in",
                type="primary",
                use_container_width=True,
            )

        if submitted:
            try:
                user = authenticate(username, password)
            except Exception:
                st.error("Could not connect to the account database. Check the hosted app's Neon connection settings.")
                return
            if user:
                st.session_state["auth_user"] = user
                st.session_state["active_page"] = LANDING_PAGE[user["role"]]
                st.rerun()
            if not get_database_connection_string():
                st.error("Login is not configured on this hosted app. Add the Neon connection string in the app's Secrets settings.")
            else:
                st.error("Username or password is incorrect.")


def render_user_management_page():
    if st.session_state["auth_user"].get("role") != "Supervisor":
        st.error("Supervisor access is required to manage users.")
        return

    connection_string = get_database_connection_string()
    st.markdown('<div class="eyebrow">ACCESS CONTROL</div>', unsafe_allow_html=True)
    st.title("User Management")
    if not connection_string:
        st.error("Configure the PostgreSQL connection before creating user accounts.")
        return

    with st.form("create_user_form"):
        username = st.text_input("Username", max_chars=80)
        password = st.text_input("Temporary password", type="password")
        role = st.selectbox("Role", ("Health Worker", "Supervisor"))
        submitted = st.form_submit_button("Create user", type="primary")

    if submitted:
        try:
            create_user_account(connection_string, username, password, role)
            st.success(f"Created {role} account for {username.strip()}.")
        except ValueError as error:
            st.error(str(error))
        except Exception:
            st.error("The account could not be created. Check the database connection and try again.")

    st.subheader("Accounts created here")
    try:
        accounts = get_user_accounts(connection_string)
    except Exception:
        st.error("User accounts could not be loaded from the database.")
        return
    if accounts:
        st.dataframe(
            [{"Username": account["username"], "Role": account["role"], "Created": account["created_at"]} for account in accounts],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No accounts have been created here yet.")


def get_database_connection_string():
    connection_string = os.getenv("DATABASE_URL")
    if connection_string:
        return connection_string
    try:
        return st.secrets["postgres"]["connection_string"]
    except (KeyError, FileNotFoundError):
        return None


@st.cache_resource
def ensure_case_schema(connection_string):
    initialize_database(connection_string)
    return True


@st.cache_data(ttl=30, show_spinner=False)
def cached_recent_cases(connection_string):
    return get_recent_cases(connection_string)


@st.cache_data(ttl=30, show_spinner=False)
def cached_overview_data(connection_string):
    return get_overview_data(connection_string)


@st.cache_data(ttl=30, show_spinner=False)
def cached_trends_data(connection_string):
    return get_trends_data(connection_string)


@st.cache_data(ttl=30, show_spinner=False)
def cached_state_history(connection_string, state):
    return get_synthetic_state_history(connection_string, state)


@st.cache_data(ttl=30, show_spinner=False)
def cached_monitor_rows(connection_string):
    return get_synthetic_monitor_rows(connection_string)


@st.cache_data(ttl=30, show_spinner=False)
def cached_alerts(connection_string, limit):
    return get_synthetic_alerts(connection_string, limit=limit)


@st.cache_data(ttl=15, show_spinner=False)
def cached_brain_status(connection_string):
    return get_brain_status(connection_string)


def render_case_registration():
    connection_string = get_database_connection_string()
    if not connection_string:
        st.error(
            "Neon is not configured. Add a PostgreSQL connection string under "
            "[postgres] in .streamlit/secrets.toml."
        )
        return

    try:
        ensure_case_schema(connection_string)
    except Exception:
        st.error(
            "Could not connect to Neon, so no case was saved. Check that the "
            "connection string is current and that your internet connection is available."
        )
        return

    st.info(
        "Case submissions are stored as suspected reports. They are not automatically "
        "confirmed cases or model-training labels."
    )

    with st.form("case_registration_form", clear_on_submit=False):
        st.subheader("Case details")
        disease = st.selectbox("Disease / condition *", DISEASES)
        if disease == "Other / not listed":
            disease = st.text_input("Specify disease / condition *", max_chars=120)

        report_col, state_col = st.columns(2)
        with report_col:
            report_date = st.date_input(
                "Report date *",
                value=date.today(),
                max_value=date.today(),
            )
        with state_col:
            state = st.selectbox("State *", NIGERIAN_STATES)

        location_col, age_col, sex_col = st.columns([1.2, 1, 1])
        with location_col:
            lga = st.text_input("Local Government Area (LGA) *", max_chars=120)
        with age_col:
            age_group = st.selectbox(
                "Age group *",
                ("Under 1", "1-4", "5-14", "15-24", "25-44", "45-64", "65+", "Unknown"),
            )
        with sex_col:
            sex = st.selectbox("Sex *", ("Female", "Male", "Other", "Unknown"))

        st.subheader("Symptoms and exposure")
        symptoms = st.multiselect(
            "Common symptoms",
            COMMON_SYMPTOMS,
            help="Type to search, or use the field below for symptoms not listed.",
        )
        other_symptoms = st.text_area("Other symptoms", max_chars=1500)
        environmental_exposure = st.text_area(
            "Environmental / exposure characteristics",
            max_chars=2000,
            help="For example, relevant animal contact, travel, or known case contact.",
        )

        st.subheader("Clinical and laboratory information")
        clinical_notes = st.text_area("Clinical notes", max_chars=4000)
        lab_col, findings_col = st.columns(2)
        with lab_col:
            laboratory_status = st.selectbox(
                "Laboratory status",
                ("Pending", "Sample collected", "Positive", "Negative", "Not tested"),
            )
        with findings_col:
            laboratory_findings = st.text_input("Laboratory findings", max_chars=1000)
        supporting_document_name = st.text_input(
            "Supporting document name (optional)",
            max_chars=255,
            help="Stage 2 stores document metadata only; it does not upload files.",
        )

        submitted = st.form_submit_button("Save suspected case", type="primary")

    if submitted:
        if disease == "Other / not listed" or not disease.strip():
            st.error("Enter a disease or condition before saving.")
        elif not lga.strip():
            st.error("Enter the Local Government Area before saving.")
        else:
            case = {
                "disease": disease.strip(),
                "report_date": report_date,
                "state": state,
                "lga": lga.strip(),
                "age_group": age_group,
                "sex": sex,
                "symptoms": symptoms,
                "other_symptoms": other_symptoms.strip(),
                "environmental_exposure": environmental_exposure.strip(),
                "clinical_notes": clinical_notes.strip(),
                "laboratory_status": laboratory_status,
                "laboratory_findings": laboratory_findings.strip(),
                "supporting_document_name": supporting_document_name.strip(),
            }
            try:
                saved = save_case(connection_string, case)
                st.cache_data.clear()
                st.session_state["saved_case_id"] = saved["id"]
            except Exception:
                st.error(
                    "The case could not be saved. No record was added. Check the "
                    "Neon connection and try again."
                )

    saved_case_id = st.session_state.pop("saved_case_id", None)
    if saved_case_id is not None:
        st.success(f"Suspected case saved successfully (record #{saved_case_id}).")

    st.divider()
    st.subheader("Recently saved cases")
    try:
        recent_cases = cached_recent_cases(connection_string)
    except Exception:
        st.warning("Recent cases could not be loaded from Neon.")
        return

    if recent_cases:
        st.dataframe(recent_cases, use_container_width=True, hide_index=True)
    else:
        st.caption("No cases have been saved yet.")


def render_overview_page():
    connection_string = get_database_connection_string()
    if not connection_string:
        st.warning("No database connection is configured for the synthetic surveillance data.")
        return

    try:
        overview, top_states = cached_overview_data(connection_string)
    except Exception:
        st.error("The synthetic surveillance dataset is unavailable from the database right now.")
        return

    st.markdown('<div class="eyebrow">SYNTHETIC DEMONSTRATION DATA</div>', unsafe_allow_html=True)
    st.title("Overview")
    st.caption("Dataset summary from the synthetic Lassa fever surveillance tables.")

    if not overview:
        st.info("No synthetic surveillance data has been loaded yet.")
        return

    cards = st.columns(4)
    cards[0].metric("Case records", f"{overview.get('total_case_records', 0):,}")
    cards[1].metric("States covered", f"{overview.get('states_covered', 0):,}")
    cards[2].metric("LGAs covered", f"{overview.get('lgas_covered', 0):,}")
    cards[3].metric("Outbreak months", f"{overview.get('outbreak_month_rows', 0):,}")

    st.subheader("Data coverage")
    coverage_col1, coverage_col2, coverage_col3 = st.columns(3)
    coverage_col1.metric("State-month rows", f"{overview.get('state_month_rows', 0):,}")
    coverage_col2.metric("First period", str(overview.get('first_period', 'n/a')))
    coverage_col3.metric("Latest period", str(overview.get('last_period', 'n/a')))

    st.subheader("Top reporting states")
    if top_states:
        st.dataframe(
            [{
                "State": row["state"],
                "Confirmed cases": int(row["total_confirmed_cases"]),
                "Outbreak months": int(row["outbreak_months"]),
                "Latest period": str(row["latest_period"]),
            } for row in top_states],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption("No synthetic state-month data is currently available.")

    st.caption("This page uses synthetic data for demonstration and must not be described as real SORMAS or NCDC observations.")


def render_trends_page():
    connection_string = get_database_connection_string()
    if not connection_string:
        st.warning("No database connection is configured for the synthetic trend analysis.")
        return

    try:
        summary, states = cached_trends_data(connection_string)
    except Exception:
        st.error("The synthetic trend data could not be loaded from Neon.")
        return

    st.markdown('<div class="eyebrow">SYNTHETIC DATA</div>', unsafe_allow_html=True)
    st.title("Trends")
    st.caption("Monthly view of confirmed Lassa fever cases and outbreak activity derived from the synthetic dataset.")

    if not summary:
        st.info("No synthetic trend data is available yet.")
        return

    state_filter = st.selectbox("Filter by state", ["All states"] + states)
    if state_filter == "All states":
        chart_data = [{"period": row["period"], "confirmed": int(row["total_confirmed"]), "outbreaks": int(row["outbreak_months"])} for row in summary]
    else:
        try:
            history = cached_state_history(connection_string, state_filter)
            chart_data = [{"period": row["period"], "confirmed": int(row["confirmed"]), "outbreaks": int(row["outbreak"])} for row in history]
        except Exception:
            st.error("The selected state could not be loaded.")
            return

    st.subheader("Confirmed cases by month")
    if chart_data:
        chart_df = __import__("pandas").DataFrame(chart_data)
        st.line_chart(chart_df.set_index("period")["confirmed"])

    st.subheader("Monthly summary")
    st.dataframe(chart_data, use_container_width=True, hide_index=True)
    st.caption("This chart reflects synthetic demonstration data only and should not be described as real epidemiological reporting.")


def render_outbreak_monitor_page():
    connection_string = get_database_connection_string()
    if not connection_string:
        st.warning("No database connection is configured for the outbreak monitor.")
        return

    try:
        monitor_rows = cached_monitor_rows(connection_string)
    except Exception:
        st.error("The synthetic outbreak monitor data could not be loaded.")
        return

    st.markdown('<div class="eyebrow">SYNTHETIC DATA</div>', unsafe_allow_html=True)
    st.title("Outbreak Monitor")
    st.caption("Latest state-level synthetic outbreak risk snapshot, ordered by recent outbreak pressure.")

    if not monitor_rows:
        st.info("No outbreak-monitor data is available.")
        return

    rows = []
    for row in monitor_rows:
        confirmed = int(row["confirmed"])
        outbreak = int(row["outbreak"])
        q70 = float(row["q70"]) if row.get("q70") is not None else 0.0
        risk_label = "HIGH" if outbreak == 1 else ("MEDIUM" if confirmed >= q70 * 0.8 else "LOW")
        rows.append({
            "State": row["state"],
            "Latest period": str(row["period"]),
            "Recent cases": confirmed,
            "Outbreak flag": outbreak,
            "70th percentile": round(q70, 2),
            "Risk": risk_label,
        })

    st.dataframe(rows, use_container_width=True, hide_index=True)
    st.caption("High-risk results are for decision support only and require epidemiological review before action.")


def render_alerts_page():
    connection_string = get_database_connection_string()
    if not connection_string:
        st.warning("No database connection is configured for the alerts view.")
        return

    try:
        alerts = cached_alerts(connection_string, 10)
    except Exception:
        st.error("The synthetic alert feed could not be loaded.")
        return

    st.markdown('<div class="eyebrow">SYNTHETIC DATA</div>', unsafe_allow_html=True)
    st.title("Alerts")
    st.caption("High- and medium-priority synthetic surveillance alerts derived from outbreak-state summaries.")

    if not alerts:
        st.info("No synthetic alerts are currently active.")
        return

    alert_rows = []
    for row in alerts:
        alert_rows.append({
            "State": row["state"],
            "Period": str(row["period"]),
            "Disease": "Lassa fever",
            "Severity": row["risk_label"],
            "Confirmed cases": int(row["confirmed"]),
            "Q70 threshold": round(float(row["q70"]), 2),
            "Message": f"Synthetic alert for {row['state']}: recent confirmed cases suggest elevated outbreak risk; review and verify before dispatch.",
            "Status": "Open",
        })

    st.dataframe(alert_rows, use_container_width=True, hide_index=True)
    st.caption("This alert list is an example decision-support view and is not a production dispatch system.")


def render_system_brain_page():
    connection_string = get_database_connection_string()
    if not connection_string:
        st.warning("No database connection is configured for the surveillance engine view.")
        return

    if st.session_state["auth_user"].get("role") != "Supervisor":
        st.error("Supervisor access is required to run the surveillance cycle.")
        return

    st.markdown('<div class="eyebrow">SYNTHETIC DATA · MAPE-K CYCLE</div>', unsafe_allow_html=True)
    st.title("System Brain")
    st.caption("Adaptive surveillance decision support. High-risk signals require human epidemiological review.")

    if st.button("Run surveillance cycle", type="primary", icon="▶"):
        try:
            run_cycle(connection_string)
            st.cache_data.clear()
            st.success("Surveillance cycle completed.")
        except Exception as error:
            st.error(f"The surveillance cycle failed: {error}")

    try:
        status = cached_brain_status(connection_string)
    except Exception:
        st.error("The surveillance engine status could not be loaded from Neon.")
        return

    last_run = status["last_run"]
    run_metrics = st.columns(4)
    run_metrics[0].metric("Last run", str(last_run["finished_at"] or last_run["started_at"]) if last_run else "NEVER")
    run_metrics[1].metric("Status", last_run["status"] if last_run else "NOT RUN")
    run_metrics[2].metric("Records processed", f"{last_run['records_processed']:,}" if last_run else "0")
    run_metrics[3].metric("Alerts generated", f"{last_run['alerts_generated']:,}" if last_run else "0")

    if last_run and last_run.get("message"):
        st.caption(last_run["message"])

    st.subheader("Latest ASVM classifications and risk levels")
    predictions = status["predictions"]
    if predictions:
        st.dataframe([{
            "State": row["state"],
            "Period": str(row["period"]),
            "Decision score": round(float(row["decision_score"]), 4),
            "Classification": "Outbreak" if float(row["decision_score"]) > 0 else "Non-outbreak",
            "Outbreak probability": round(float(row["probability"]), 4) if row["probability"] is not None else "Uncalibrated",
            "Risk": row["risk_category"],
            "Observed target": int(row["actual_outbreak"]),
        } for row in predictions], use_container_width=True, hide_index=True)
    else:
        st.info("No ASVM classifications yet. Run the cycle to initialize and process the synthetic data.")

    st.subheader("Generated alerts")
    alerts = status["alerts"]
    if alerts:
        st.dataframe([{
            "State": row["state"],
            "Disease": row["disease"],
            "Period": str(row["period"]),
            "Probability": round(float(row["probability"]), 4),
            "Risk": row["risk_category"],
            "Created": str(row["created_at"]),
        } for row in alerts], use_container_width=True, hide_index=True)
    else:
        st.caption("No ASVM-generated HIGH-risk alerts are recorded.")

    st.subheader("Run history")
    if status["runs"]:
        st.dataframe([{
            "Run": row["id"],
            "Started": str(row["started_at"]),
            "Finished": str(row["finished_at"]) if row["finished_at"] else "Running",
            "Status": row["status"],
            "Processed": int(row["records_processed"]),
            "Classifications": int(row["predictions_made"]),
            "Alerts": int(row["alerts_generated"]),
        } for row in status["runs"]], use_container_width=True, hide_index=True)
    st.caption("Synthetic demonstration data only. Alerts are investigation signals, not outbreak confirmation or automatic dispatch.")


def render_model_page():
    connection_string = get_database_connection_string()
    if not connection_string:
        st.warning("No database connection is configured for model data inspection.")
        return

    try:
        overview, _ = cached_overview_data(connection_string)
        status = cached_brain_status(connection_string)
    except Exception:
        st.error("Model inputs could not be read from the synthetic dataset.")
        return

    st.markdown('<div class="eyebrow">ASVM MODEL WORKSPACE</div>', unsafe_allow_html=True)
    st.title("Model")
    st.caption("Adaptive linear SVM configuration and prequential evaluation on synthetic data.")

    model_ready = status.get(
        "model_ready",
        bool(status.get("model") and status["model"].get("model_version") == MODEL_VERSION),
    )
    model_status = "TRAINED" if model_ready else "NOT FITTED"
    status_col, records_col, states_col, scored_col = st.columns(4)
    status_col.metric("Training status", model_status)
    records_col.metric("Synthetic records", f"{overview.get('total_case_records', 0):,}")
    states_col.metric("State-month samples", f"{overview.get('state_month_rows', 0):,}")
    scored_col.metric("Evaluated classifications", f"{status['evaluation']['evaluated']:,}")

    st.subheader("Model configuration")
    st.dataframe([
        {"Setting": "Estimator", "Value": "SGDClassifier (linear SVM)"},
        {"Setting": "Loss / regularisation", "Value": "hinge / L2, alpha = 1e-4"},
        {"Setting": "Class balance", "Value": "Balanced weights from warm-up labels"},
        {"Setting": "Incremental update", "Value": "partial_fit()"},
        {"Setting": "Calibration", "Value": "Platt scaling fitted on warm-up margins"},
        {"Setting": "Evaluation", "Value": "15% chronological warm-up, then test-then-train batches of 100"},
        {"Setting": "Classification rule", "Value": "Outbreak when SVM decision score > 0"},
        {"Setting": "Risk rule", "Value": "HIGH >= 0.50; MEDIUM >= 0.20; LOW < 0.20"},
    ], use_container_width=True, hide_index=True)

    st.subheader("Features and target")
    feature_rows = [
        {"Role": "Target", "Field": "outbreak", "Meaning": "Confirmed cases exceed the state’s historical 70th percentile and are at least 3"},
        {"Role": "Candidate feature", "Field": "lag1", "Meaning": "Confirmed cases in the previous month"},
        {"Role": "Candidate feature", "Field": "lag2", "Meaning": "Confirmed cases two months earlier"},
        {"Role": "Candidate feature", "Field": "roll3", "Meaning": "Three-month rolling mean"},
        {"Role": "Candidate feature", "Field": "trend", "Meaning": "lag1 minus lag2"},
        {"Role": "Candidate feature", "Field": "hotspot", "Meaning": "Historical hotspot indicator (0/1)"},
        {"Role": "Candidate feature", "Field": "sin_m, cos_m", "Meaning": "Seasonal encoding of calendar month"},
    ]
    st.dataframe(feature_rows, use_container_width=True, hide_index=True)

    evaluation = status["evaluation"]
    evaluated = int(evaluation["evaluated"] or 0)
    if evaluated:
        true_positive = int(evaluation["true_positive"] or 0)
        false_positive = int(evaluation["false_positive"] or 0)
        false_negative = int(evaluation["false_negative"] or 0)
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        st.subheader("Prequential classification evaluation")
        st.caption(f"Observed metrics from {evaluated:,} pre-update SVM classifications; outbreak class is assigned when decision score > 0. Probability thresholds are used only for risk levels.")
        metric_cols = st.columns(3)
        metric_cols[0].metric("Precision", f"{precision:.3f}")
        metric_cols[1].metric("Outbreak recall", f"{recall:.3f}")
        metric_cols[2].metric("F1 score", f"{f1:.3f}")
        st.dataframe([
            {"Actual / classification": "Outbreak", "Classified outbreak": true_positive, "Classified non-outbreak": false_negative},
            {"Actual / classification": "Non-outbreak", "Classified outbreak": false_positive, "Classified non-outbreak": int(evaluation["true_negative"] or 0)},
        ], use_container_width=True, hide_index=True)
    else:
        st.info("Run the System Brain cycle to train the model and collect out-of-sample classifications. No evaluation metrics are available yet.")

    baseline = status["baseline"]
    st.subheader("LinearSVC temporal baseline")
    if baseline:
        st.caption(f"Fit once on the first {baseline['train_rows']:,} observations; evaluated on {baseline['test_rows']:,} later observations beginning {baseline['test_start']}. Classification uses decision score > 0.")
        baseline_metrics = st.columns(3)
        baseline_metrics[0].metric("Precision", f"{float(baseline['precision']):.3f}")
        baseline_metrics[1].metric("Outbreak recall", f"{float(baseline['recall']):.3f}")
        baseline_metrics[2].metric("F1 score", f"{float(baseline['f1']):.3f}")
        comparison = []
        if evaluated:
            comparison.append({"Model": "Adaptive SVM (prequential)", "Test classifications": evaluated, "Precision": f"{precision:.3f}", "Recall": f"{recall:.3f}", "F1": f"{f1:.3f}"})
        comparison.append({"Model": "LinearSVC (70/30 temporal)", "Test classifications": int(baseline["test_rows"]), "Precision": f"{float(baseline['precision']):.3f}", "Recall": f"{float(baseline['recall']):.3f}", "F1": f"{float(baseline['f1']):.3f}"})
        st.dataframe(comparison, use_container_width=True, hide_index=True)
    else:
        st.info("The LinearSVC baseline will be created with the next System Brain cycle.")

    st.caption("All records shown here are synthetic demonstration data. Classification metrics do not establish real-world clinical performance.")


def render_page(page):
    if page == "User Management":
        render_user_management_page()
        return
    if page == "Register a Case":
        render_case_registration()
        return
    if page == "Overview":
        render_overview_page()
        return
    if page == "Trends":
        render_trends_page()
        return
    if page == "Outbreak Monitor":
        render_outbreak_monitor_page()
        return
    if page == "Alerts":
        render_alerts_page()
        return
    if page == "System Brain":
        render_system_brain_page()
        return
    if page == "Model":
        render_model_page()
        return

    st.markdown('<div class="eyebrow">SURVEILLANCE WORKSPACE</div>', unsafe_allow_html=True)
    st.title(page)
    st.caption(PAGES[page])
    st.markdown(
        '<div class="empty-state"><strong>Reserved for a later stage.</strong><br>'
        'This page remains an empty shell in Stage 2. Trends, outbreak monitoring, '
        'alerts, and model outputs are not connected yet.</div>',
        unsafe_allow_html=True,
    )


DEMO_USERS = load_demo_users()
add_styles()

if "auth_user" not in st.session_state:
    render_login()
    st.stop()

user = st.session_state["auth_user"]
role = user.get("role")
allowed_pages = ROLE_PAGES.get(role, ())
if not allowed_pages:
    st.session_state.pop("auth_user", None)
    st.rerun()

active_page = st.session_state.get("active_page", LANDING_PAGE[role])
if active_page not in allowed_pages:
    active_page = LANDING_PAGE[role]
    st.session_state["active_page"] = active_page

with st.sidebar:
    st.markdown(
        """
        <div class="brand-lockup">
            <div class="brand-mark">+</div>
            <div>
                <div class="brand-name">Disease Surveillance</div>
                <div class="brand-subtitle">Local demonstration</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(f'<span class="role-chip">{role}</span>', unsafe_allow_html=True)
    st.divider()
    for page in allowed_pages:
        button_type = "primary" if page == active_page else "secondary"
        if st.button(page, key=f"nav_{page}", use_container_width=True, type=button_type):
            st.session_state["active_page"] = page
            st.rerun()
    st.divider()
    st.caption(f"Signed in as {user['username']}")
    if st.button("Sign out", use_container_width=True):
        st.session_state.pop("auth_user", None)
        st.session_state.pop("active_page", None)
        st.session_state.pop("sidebar_page", None)
        st.rerun()

if st.session_state.get("active_page") not in ROLE_PAGES.get(user.get("role"), ()):
    st.error("You do not have permission to view this page.")
    st.stop()

render_page(st.session_state["active_page"])
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import date, timedelta
import uuid

# ------------------ Helpers ------------------

FREQ_MAP = {
    "Once": ("once", 1),
    "Daily": ("days", 1),
    "Weekly": ("weeks", 1),
    "Fortnightly": ("weeks", 2),
    "4-weekly": ("weeks", 4),
    "Monthly": ("months", 1),
    "Quarterly": ("months", 3),
    "6-monthly": ("months", 6),
    "Annually": ("years", 1),
}

def freq_label_from_unit_interval(unit, interval):
    for label, (u, i) in FREQ_MAP.items():
        if u == unit and int(i) == int(interval):
            return label
    return f"{unit} x{interval}"

def safe_date_parse(d):
    if d is None or d == "":
        return None
    if isinstance(d, date):
        return d
    try:
        return pd.to_datetime(d).date()
    except:
        return None

def normalize_amount(a):
    try:
        return float(a)
    except:
        return None

def shift_to_business_day(d):
    while d.weekday() >= 5:  # Sat=5, Sun=6
        d = d + timedelta(days=1)
    return d

def generate_occurrences(item):
    occurrences = []
    amt = normalize_amount(item.get("amount"))
    if amt is None:
        return occurrences
    amt = -abs(amt) if item.get("direction") == "expense" else abs(amt)

    start = safe_date_parse(item.get("start_date"))
    if start is None:
        return occurrences

    end = safe_date_parse(item.get("end_date"))
    unit = item.get("freq_unit", "once")
    interval = int(item.get("interval", 1))

    if unit == "once":
        occurrences.append((shift_to_business_day(start), amt, item.get("account")))
        return occurrences

    current = start
    max_iter = 5000
    it = 0

    while it < max_iter:
        current_shifted = shift_to_business_day(current)
        occurrences.append((current_shifted, amt, item.get("account")))

        if unit == "days":
            current = current + timedelta(days=interval)
        elif unit == "weeks":
            current = current + timedelta(weeks=interval)
        elif unit == "months":
            nd = pd.date_range(start=current, periods=2, freq=f"{interval}M")[1].date()
            current = nd
        elif unit == "years":
            nd = pd.date_range(start=current, periods=2, freq=f"{interval}Y")[1].date()
            current = nd
        else:
            break

        it += 1
        if end and current > end:
            break

    return occurrences

def build_forecast(items, accounts, start_date, days, threshold):
    if not accounts:
        return None, pd.DataFrame()

    all_dates = pd.date_range(start=start_date, periods=days).date
    balances = {a["id"]: [a["opening"]] * days for a in accounts}
    account_names = {a["id"]: a["name"] for a in accounts}

    # Collect occurrences
    events = []
    for it in items:
        for d, amt, acc in generate_occurrences(it):
            if start_date <= d <= all_dates[-1]:
                events.append((d, amt, acc))

    events.sort(key=lambda x: x[0])

    for i, d in enumerate(all_dates):
        if i > 0:
            for acc in balances:
                balances[acc][i] = balances[acc][i - 1]
        for ev in [e for e in events if e[0] == d]:
            balances[ev[2]][i] += ev[1]

    df = pd.DataFrame({"date": all_dates})
    for acc in balances:
        df[account_names[acc]] = balances[acc]
    df["Total"] = df.drop(columns="date").sum(axis=1)

    low_flag = (df["Total"] < threshold).any()

    return df, events, low_flag

# ------------------ Session state ------------------

if "items" not in st.session_state:
    st.session_state["items"] = []

if "accounts" not in st.session_state:
    st.session_state["accounts"] = []

# ------------------ UI ------------------

st.title("💰 Money Tracker & Forecast")

# ---- Account Management ----
st.header("Accounts")
col1, col2 = st.columns(2)
with col1:
    acc_name = st.text_input("Account name", key="acc_name")
    opening = st.number_input("Opening balance", value=0.0, step=100.0, key="acc_opening")
    if st.button("➕ Add Account"):
        if acc_name:
            st.session_state["accounts"].append({"id": str(uuid.uuid4()), "name": acc_name, "opening": opening})
            st.success(f"Added account: {acc_name}")
            st.experimental_rerun()
with col2:
    if st.session_state["accounts"]:
        to_remove = st.selectbox("Remove account", options=[a["name"] for a in st.session_state["accounts"]])
        if st.button("🗑️ Remove"):
            st.session_state["accounts"] = [a for a in st.session_state["accounts"] if a["name"] != to_remove]
            st.success(f"Removed account: {to_remove}")
            st.experimental_rerun()

if st.session_state["accounts"]:
    st.table(pd.DataFrame(st.session_state["accounts"])[["name", "opening"]])
else:
    st.info("No accounts yet. Add at least one.")

# ---- Planned Items ----
st.header("Planned Income & Expenses")
with st.form("item_form", clear_on_submit=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        name = st.text_input("Name")
        direction = st.selectbox("Type", ["income", "expense"])
        amt_val = st.number_input("Amount", step=10.0)
    with c2:
        freq_label = st.selectbox("Frequency", list(FREQ_MAP.keys()))
        start_date = st.date_input("Start date", value=date.today())
        end_date = st.date_input("End date (optional)", value=None)
    with c3:
        if st.session_state["accounts"]:
            acc_choice = st.selectbox("Account", [a["name"] for a in st.session_state["accounts"]])
        else:
            acc_choice = None
        notes = st.text_area("Notes")

    submitted = st.form_submit_button("Add Item")
    if submitted:
        if not acc_choice:
            st.error("Add an account first.")
        else:
            unit, interval = FREQ_MAP[freq_label]
            acc_id = next(a["id"] for a in st.session_state["accounts"] if a["name"] == acc_choice)
            st.session_state["items"].append({
                "name": name,
                "direction": direction,
                "amount": amt_val,
                "freq_unit": unit,
                "interval": interval,
                "frequency_label": freq_label,
                "start_date": start_date,
                "end_date": end_date,
                "notes": notes,
                "account": acc_id,
            })
            st.success("Item added!")

if st.session_state["items"]:
    st.dataframe(pd.DataFrame(st.session_state["items"])[
        ["name","direction","amount","frequency_label","account","start_date","end_date","notes"]
    ])

# ---- Forecast ----
st.header("Forecast")
forecast_days = 365
threshold = st.number_input("Balance alert threshold", value=0.0, step=100.0)

if st.button("Generate Forecast"):
    df, events, low_flag = build_forecast(st.session_state["items"], st.session_state["accounts"], date.today(), forecast_days, threshold)

    if df is not None and not df.empty:
        st.subheader("Daily Balances")
        st.dataframe(df)

        # Calendar heat map (total balance)
        df_heat = df.copy()
        df_heat["month"] = pd.to_datetime(df_heat["date"]).dt.to_period("M").astype(str)
        fig = px.density_heatmap(df_heat, x="date", y="month", z="Total", color_continuous_scale="RdYlGn",
                                 title="Balance Heatmap (Total)")
        st.plotly_chart(fig, use_container_width=True)

        if low_flag:
            st.error("⚠️ Forecast dips below threshold!")
        else:
            st.success("✅ Balance stays above threshold.")
    else:
        st.error("Forecast generation failed. Add accounts and items first.")

# ---- Google Sheets Integration (placeholder) ----
st.header("Export / Backup")
if st.button("Export to Google Sheets"):
    st.info("Google Sheets export not configured. Add service account JSON + gspread setup.")

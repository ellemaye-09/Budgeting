import streamlit as st
import pandas as pd
from datetime import date, timedelta
import calendar

# --------------------------
# Utility functions
# --------------------------

def safe_date_parse(d):
    if not d:
        return None
    if isinstance(d, date):
        return d
    try:
        return pd.to_datetime(d).date()
    except Exception:
        return None

def normalize_amount(val):
    try:
        return float(val)
    except Exception:
        return None

def generate_occurrences(item, horizon_days=365):
    """Generate all occurrences for one item, based on chosen frequency."""
    occurrences = []
    amt = normalize_amount(item.get("amount"))
    if amt is None:
        return occurrences

    amt = -abs(amt) if item.get("direction", "expense") == "expense" else abs(amt)

    start = safe_date_parse(item.get("start_date"))
    if not start:
        return occurrences

    end = safe_date_parse(item.get("end_date")) or (date.today() + timedelta(days=horizon_days))
    freq = item.get("frequency", "once")

    current = start
    while current <= end:
        occurrences.append((current, amt))

        if freq == "daily":
            current += timedelta(days=1)
        elif freq == "weekly":
            current += timedelta(weeks=1)
        elif freq == "fortnightly":
            current += timedelta(weeks=2)
        elif freq == "4-weekly":
            current += timedelta(weeks=4)
        elif freq == "monthly":
            month = current.month + 1
            year = current.year
            if month > 12:
                month = 1
                year += 1
            day = min(current.day, calendar.monthrange(year, month)[1])
            current = date(year, month, day)
        elif freq == "quarterly":
            month = current.month + 3
            year = current.year
            if month > 12:
                month -= 12
                year += 1
            day = min(current.day, calendar.monthrange(year, month)[1])
            current = date(year, month, day)
        elif freq == "6-monthly":
            month = current.month + 6
            year = current.year
            if month > 12:
                month -= 12
                year += 1
            day = min(current.day, calendar.monthrange(year, month)[1])
            current = date(year, month, day)
        elif freq == "annually":
            try:
                current = date(current.year + 1, current.month, current.day)
            except:
                current = date(current.year + 1, current.month, 28)
        else:  # once
            break

    return occurrences


def build_forecast(items, accounts, days=365, min_balance=0):
    """Build a day-by-day forecast for all accounts."""
    today = date.today()
    horizon = today + timedelta(days=days)
    all_dates = pd.date_range(today, horizon)

    balances = pd.DataFrame(index=all_dates, columns=accounts.keys())
    balances = balances.fillna(0.0)

    # Initialise with opening balances
    for acc, opening in accounts.items():
        balances.loc[:, acc] = opening

    # Apply each item to forecast
    for item in items:
        acc = item.get("account")
        if acc not in accounts:
            continue
        occurrences = generate_occurrences(item, horizon_days=days)
        for d, amt in occurrences:
            if d in balances.index:
                balances.loc[d:, acc] += amt  # apply forward

    # Flag low balance
    flags = (balances < min_balance)

    return balances, flags


def calendar_heatmap(balances, account):
    """Return a calendar heatmap dataframe for one account."""
    df = balances[[account]].copy()
    df["date"] = df.index.date
    df["year"] = df.index.year
    df["month"] = df.index.month
    df["day"] = df.index.day

    pivot = df.pivot_table(index=["year", "month"], columns="day", values=account, aggfunc="last")
    return pivot


# --------------------------
# Streamlit App
# --------------------------

st.set_page_config(page_title="Money Forecast", layout="wide")

st.title("📊 Money Tracker & Forecast")

# Session state init
if "items" not in st.session_state:
    st.session_state["items"] = []
if "accounts" not in st.session_state:
    st.session_state["accounts"] = {}

# Sidebar for account management
st.sidebar.header("Accounts")
with st.sidebar:
    acc_name = st.text_input("Add account name")
    acc_opening = st.number_input("Opening balance", value=0.0, step=100.0)
    if st.button("➕ Add account"):
        if acc_name and acc_name not in st.session_state["accounts"]:
            st.session_state["accounts"][acc_name] = acc_opening

    if st.session_state["accounts"]:
        del_acc = st.selectbox("Remove account", options=[""] + list(st.session_state["accounts"].keys()))
        if st.button("🗑️ Remove selected account") and del_acc:
            st.session_state["accounts"].pop(del_acc, None)

st.write("### Current Accounts")
st.write(st.session_state["accounts"])

# Add income/expense items
st.header("Add Planned Item")
with st.form("add_item"):
    name = st.text_input("Name")
    direction = st.selectbox("Type", ["income", "expense"])
    amount = st.number_input("Amount", value=0.0, step=10.0)
    frequency = st.selectbox("Frequency", ["once", "daily", "weekly", "fortnightly", "4-weekly", "monthly", "quarterly", "6-monthly", "annually"])
    start_date = st.date_input("Start date", value=date.today())
    end_date = st.date_input("End date (optional)", value=None)
    account = st.selectbox("Account", options=list(st.session_state["accounts"].keys()) if st.session_state["accounts"] else [])
    notes = st.text_area("Notes", "")
    submitted = st.form_submit_button("Add Item")
    if submitted and account:
        st.session_state["items"].append({
            "name": name,
            "direction": direction,
            "amount": amount,
            "frequency": frequency,
            "start_date": start_date,
            "end_date": end_date,
            "account": account,
            "notes": notes
        })
        st.success(f"Added {direction}: {name} to {account}")

# Forecast
st.header("Forecast")
min_balance = st.number_input("Minimum balance warning threshold", value=0.0, step=100.0)
days = st.slider("Forecast horizon (days)", 30, 730, 365)

if st.button("Generate Forecast"):
    if not st.session_state["accounts"]:
        st.error("Please add at least one account.")
    else:
        balances, flags = build_forecast(st.session_state["items"], st.session_state["accounts"], days=days, min_balance=min_balance)
        st.success("Forecast generated!")

        st.subheader("Balances")
        st.dataframe(balances.style.applymap(lambda v: "background-color: pink" if v < min_balance else ""))

        # Calendar-style heatmaps
        st.subheader("Calendar Heatmaps")
        for acc in st.session_state["accounts"].keys():
            st.markdown(f"**{acc}**")
            pivot = calendar_heatmap(balances, acc)
            st.dataframe(pivot.style.background_gradient(cmap="RdYlGn"))

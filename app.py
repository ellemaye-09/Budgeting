# app.py
"""
Budgeting Streamlit app
Features:
- Safe session-state handling
- Multiple accounts (7)
- Custom recurrence (Every X days/weeks/months/years)
- Business-day shift (weekend -> next Monday)
- 365-day (configurable) forecast per account + combined
- Flags days below threshold
- Calendar heatmap (matplotlib)
- Google Sheets integration (optional) using gspread + service account JSON
- CSV export
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import io
import math

# Attempt optional import for Google Sheets
try:
    import gspread
    from gspread_dataframe import set_with_dataframe, get_as_dataframe
    GS_AVAILABLE = True
except Exception:
    GS_AVAILABLE = False

# -----------------------
# Helpers
# -----------------------

def ensure_session_state_key(key, default):
    if key not in st.session_state:
        st.session_state[key] = default

def safe_date_parse(v):
    if isinstance(v, (pd.Timestamp, datetime, date)):
        return pd.to_datetime(v).date()
    try:
        return pd.to_datetime(v).date()
    except Exception:
        return None

def normalize_amount(x):
    try:
        return float(x)
    except Exception:
        return None

def shift_to_business_day(d):
    # shift saturday or sunday forward to Monday
    if not isinstance(d, date):
        d = safe_date_parse(d)
        if d is None:
            return None
    while d.weekday() >= 5:  # 5=Sat,6=Sun
        d = d + timedelta(days=1)
    return d

def add_months(orig_date, months):
    # add months preserving day where possible
    y, m = orig_date.year, orig_date.month
    ndm = m + months
    ny = y + (ndm - 1) // 12
    ndm = ((ndm - 1) % 12) + 1
    day = orig_date.day
    last_day = (pd.Timestamp(ny, ndm, 1) + pd.offsets.MonthEnd(0)).day
    day = min(day, last_day)
    return date(ny, ndm, day)

def add_years(orig_date, years):
    try:
        return date(orig_date.year + years, orig_date.month, orig_date.day)
    except Exception:
        # fallback
        return orig_date + timedelta(days=365 * years)

def generate_occurrences(item, horizon_start, horizon_end):
    """
    item: dict with keys:
      - name, amount, direction ('expense'/'income'), freq_unit ('once','days','weeks','months','years'), interval (int),
      - start_date (date), end_date (date or None), shift_business (bool), account (int)
    returns list of (date, amount_signed)
    """
    occs = []
    amt = normalize_amount(item.get("amount"))
    if amt is None:
        return occs

    amt = -abs(amt) if item.get("direction", "expense") == "expense" else abs(amt)

    start = safe_date_parse(item.get("start_date"))
    if start is None:
        return occs

    end = safe_date_parse(item.get("end_date")) if item.get("end_date") else None
    unit = item.get("freq_unit", "once")
    interval = int(item.get("interval", 1))
    shift = bool(item.get("shift_business", True))

    # Helper: check within horizon and optional end
    def in_window(d):
        return d >= horizon_start and d <= horizon_end and (end is None or d <= end)

    if unit == "once":
        d = shift_to_business_day(start) if shift else start
        if in_window(d):
            occs.append((d, amt))
        return occs

    current = start
    max_iter = 2000
    it = 0
    while it < max_iter:
        d = shift_to_business_day(current) if shift else current
        if in_window(d):
            occs.append((d, amt))
        # move forward
        if unit == "days":
            current = current + timedelta(days=interval)
        elif unit == "weeks":
            current = current + timedelta(weeks=interval)
        elif unit == "months":
            current = add_months(current, interval)
        elif unit == "years":
            current = add_years(current, interval)
        else:
            break
        it += 1
        if current > horizon_end:
            break

    return occs

def build_forecast(accounts, items, start_date, days, low_threshold_per_account):
    """
    accounts: list of dicts with keys 'name' and 'starting_balance'
    items: list of item dicts (each has 'account' index)
    returns:
      - per_account_dfs: list of DataFrames with columns date,daily_change,balance,flag_low
      - combined_df: DataFrame with combined daily_change and combined balance
    """
    start_date = safe_date_parse(start_date)
    if start_date is None:
        raise ValueError("Invalid start date for forecast.")
    horizon = pd.date_range(start=start_date, periods=days, freq="D").date
    horizon_start = horizon[0]
    horizon_end = horizon[-1]

    # per-account daily changes
    account_changes = [ {d: 0.0 for d in horizon} for _ in accounts ]

    # populate
    for it in items:
        acct_idx = int(it.get("account", 0))
        if acct_idx < 0 or acct_idx >= len(accounts):
            continue
        occs = generate_occurrences(it, horizon_start, horizon_end)
        for d, amt in occs:
            if d in account_changes[acct_idx]:
                account_changes[acct_idx][d] += amt

    # build per-account dfs
    per_account_dfs = []
    for i, acct in enumerate(accounts):
        df = pd.DataFrame([{"date": d, "daily_change": account_changes[i][d]} for d in sorted(account_changes[i].keys())])
        df["balance"] = df["daily_change"].cumsum() + float(acct.get("starting_balance", 0.0) or 0.0)
        threshold = low_threshold_per_account.get(i, None)
        if threshold is not None:
            df["flag_low"] = df["balance"] < float(threshold)
        else:
            df["flag_low"] = False
        per_account_dfs.append(df)

    # combined
    combined = pd.DataFrame({"date": horizon})
    combined["daily_change"] = 0.0
    for df in per_account_dfs:
        combined = combined.merge(df[["date", "daily_change"]], on="date", how="left", suffixes=("", "_tmp"))
        combined["daily_change"] = combined["daily_change"].fillna(0) + combined.get("daily_change_tmp", 0).fillna(0)
        if "daily_change_tmp" in combined.columns:
            combined = combined.drop(columns=["daily_change_tmp"])
    combined["balance"] = combined["daily_change"].cumsum() + sum([float(a.get("starting_balance",0) or 0) for a in accounts])
    combined["flag_low"] = False
    # also set combined flag if any account below its threshold on that date
    for i, df in enumerate(per_account_dfs):
        col_name = f"acct_{i}_bal"
        combined = combined.merge(df[["date","balance","flag_low"]].rename(columns={"balance": col_name, "flag_low": f"acct_{i}_flag"}), on="date", how="left")
        combined[col_name] = combined[col_name].fillna(method="ffill").fillna(0)
        combined[f"acct_{i}_flag"] = combined[f"acct_{i}_flag"].fillna(False)

    # combined flag if any account flag true OR combined balance < global threshold (handled outside)
    combined["any_account_flag"] = combined[[c for c in combined.columns if c.endswith("_flag")]].any(axis=1)
    return per_account_dfs, combined

# -----------------------
# App UI
# -----------------------

st.set_page_config(page_title="Budgeting — 365-day forecast", layout="wide")
st.title("Planned incomes & expenses — 365-day forecast (NZD)")
st.subheader("Multiple accounts, custom recurrence, business-day shift, calendar heatmap, Google Sheets integration (optional).")

# Initialise session state safely
ensure_session_state_key("items", [])  # list of item dicts
ensure_session_state_key("accounts", [])  # list of account dicts
ensure_session_state_key("low_thresholds", {})  # dict acct_idx -> threshold

# Setup default 7 accounts if not present
if not st.session_state["accounts"]:
    default_accounts = []
    for i in range(7):
        default_accounts.append({"name": f"Account {i+1}", "starting_balance": 0.0})
    st.session_state["accounts"] = default_accounts

# Sidebar global settings
with st.sidebar:
    st.header("Settings")
    today = date.today()
    forecast_start = st.date_input("Forecast start date", value=today)
    days = st.number_input("Forecast length (days)", min_value=30, max_value=365*2, value=365)
    global_low_threshold = st.number_input("Global low threshold (flag if combined balance < ) (NZD)", value=-500.0, step=50.0, format="%.2f")
    st.markdown("Google Sheets integration (optional): upload service account JSON below or set environment variable on your host.")
    gs_json = st.file_uploader("Upload Google service account JSON (optional)", type=["json"])
    st.markdown("If you want auto-save to Google Sheets, upload the JSON and enter a Sheet name below.")
    gs_sheet_name = st.text_input("Google Sheet name (optional)")

# Accounts management
st.header("Accounts (7)")
acct_cols = st.columns([3,2,2,1])
for i, acct in enumerate(st.session_state["accounts"]):
    with acct_cols[0]:
        name = st.text_input(f"Account {i+1} name", value=acct.get("name",""), key=f"acct_name_{i}")
        st.session_state["accounts"][i]["name"] = name
    with acct_cols[1]:
        sb = st.number_input(f"Account {i+1} starting balance (NZD)", value=float(acct.get("starting_balance",0.0)), key=f"acct_start_{i}", format="%.2f")
        st.session_state["accounts"][i]["starting_balance"] = sb
    with acct_cols[2]:
        lt = st.number_input(f"Low threshold (acct {i+1})", value=float(st.session_state["low_thresholds"].get(i, -1000.0)), key=f"acct_thresh_{i}", format="%.2f")
        st.session_state["low_thresholds"][i] = lt
    with acct_cols[3]:
        st.markdown(f"**#{i}**")

# Add / edit items
st.markdown("---")
st.header("Add or edit a planned item (income/expense)")
with st.form("item_form", clear_on_submit=False):
    c1, c2, c3 = st.columns([3,2,2])
    with c1:
        name = st.text_input("Name (e.g. Mortgage, Salary)", key="input_name")
        direction = st.selectbox("Type", options=["expense","income"], index=0, key="input_direction", format_func=lambda x: "Expense" if x=="expense" else "Income")
        notes = st.text_area("Notes (optional)", key="input_notes", height=60)
    with c2:
        amount = st.text_input("Amount (NZD) — positive number", key="input_amount")
        start_date = st.date_input("Start date", value=today, key="input_start")
        end_date = st.date_input("End date (optional)", value=None, key="input_end")
    with c3:
        freq_unit = st.selectbox("Frequency unit", options=["once","days","weeks","months","years"], key="input_freq_unit")
        interval = 1
        if freq_unit != "once":
            interval = st.number_input(f"Every how many {freq_unit}?", min_value=1, value=1, step=1, key="input_interval")
        account = st.selectbox("Account", options=list(range(len(st.session_state["accounts"]))), format_func=lambda x: st.session_state["accounts"][x]["name"], key="input_account")
        shift_business = st.checkbox("Shift to next Monday if on weekend", value=True, key="input_shift")

    submitted = st.form_submit_button("Add / Update item")

if submitted:
    errors = []
    if not name or str(name).strip()=="":
        errors.append("Name required.")
    amt_val = normalize_amount(amount)
    if amt_val is None:
        errors.append("Amount must be a number.")
    if safe_date_parse(start_date) is None:
        errors.append("Invalid start date.")
    if end_date:
        if safe_date_parse(end_date) is None:
            errors.append("Invalid end date.")
        elif safe_date_parse(end_date) < safe_date_parse(start_date):
            errors.append("End date cannot be before start date.")
    if errors:
        for e in errors:
            st.error(e)
    else:
        new_item = {
            "name": name.strip(),
            "direction": direction,
            "amount": amt_val,
            "freq_unit": freq_unit,
            "interval": int(interval),
            "start_date": safe_date_parse(start_date),
            "end_date": safe_date_parse(end_date) if end_date else None,
            "notes": notes,
            "shift_business": bool(shift_business),
            "account": int(account),
        }
        # append
        st.session_state["items"].append(new_item)
        st.success("Item added to planned items.")

# Display current items
st.markdown("---")
st.header("Current planned items")
items = st.session_state.get("items", [])
if not isinstance(items, list):
    st.error("Internal error: items not a list. Resetting.")
    st.session_state["items"] = []
    items = st.session_state["items"]

if items:
    try:
        df_display = pd.DataFrame(items)
        # format dates
        if "start_date" in df_display.columns:
            df_display["start_date"] = df_display["start_date"].apply(lambda x: safe_date_parse(x) or "")
        if "end_date" in df_display.columns:
            df_display["end_date"] = df_display["end_date"].apply(lambda x: safe_date_parse(x) or "")
        # map account number to name
        if "account" in df_display.columns:
            df_display["account_name"] = df_display["account"].apply(lambda x: st.session_state["accounts"][int(x)]["name"] if (isinstance(x,int) and 0<=x<len(st.session_state["accounts"])) else str(x))
        st.dataframe(df_display[["name","direction","amount","freq_unit","interval","account_name","start_date","end_date","notes"]])
    except Exception as e:
        st.error(f"Failed to render items: {e}")
else:
    st.info("No planned items yet.")

# Edit / remove controls
if items:
    st.markdown("---")
    st.subheader("Edit / Remove item")
    idx = st.number_input("Index (0-based) to edit/remove", min_value=0, max_value=max(0, len(items)-1), value=0, step=1)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Load for edit"):
            it = st.session_state["items"][int(idx)]
            # populate the form inputs via session_state keys
            st.session_state["input_name"] = it.get("name","")
            st.session_state["input_direction"] = it.get("direction","expense")
            st.session_state["input_amount"] = str(it.get("amount",""))
            st.session_state["input_freq_unit"] = it.get("freq_unit","once")
            st.session_state["input_interval"] = int(it.get("interval",1))
            st.session_state["input_account"] = int(it.get("account",0))
            st.session_state["input_start"] = it.get("start_date", today)
            st.session_state["input_end"] = it.get("end_date", None)
            st.session_state["input_shift"] = bool(it.get("shift_business", True))
            st.session_state["input_notes"] = it.get("notes","")
            # Save index for update
            st.session_state["edit_index"] = int(idx)
            st.experimental_rerun()
    with col2:
        if st.button("Remove item"):
            removed = st.session_state["items"].pop(int(idx))
            st.success(f"Removed: {removed.get('name','(unknown)')}")
            st.experimental_rerun()

# If edit index exists and form resubmitted, update instead of append
if submitted and st.session_state.get("edit_index", None) is not None:
    ei = st.session_state["edit_index"]
    if 0 <= ei < len(st.session_state["items"]):
        # replace last appended (we appended earlier). To be safe, replace by index.
        st.session_state["items"][ei] = st.session_state["items"][-1]
        st.session_state["items"].pop()  # remove the duplicate last
        st.success("Item updated.")
        del st.session_state["edit_index"]

# Forecast generation
st.markdown("---")
st.header("Forecast results")

try:
    per_acct_dfs, combined = build_forecast(st.session_state["accounts"], st.session_state["items"], forecast_start, int(days), st.session_state["low_thresholds"])
    # apply global low threshold
    combined["global_flag"] = combined["balance"] < float(global_low_threshold)

    # show metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Forecast start", f"{forecast_start}")
    min_bal = combined["balance"].min()
    min_date = combined.loc[combined["balance"].idxmin(),"date"]
    col2.metric("Minimum combined balance (NZD)", f"{min_bal:,.2f}", delta=None)
    flagged_days = int(combined["global_flag"].sum() or 0)
    col3.metric("Days combined below global threshold", f"{flagged_days}")

    # show per-account min balances
    acct_mins = []
    for i, df in enumerate(per_acct_dfs):
        m = df["balance"].min()
        d = df.loc[df["balance"].idxmin(),"date"]
        acct_mins.append((st.session_state["accounts"][i]["name"], m, d))
    st.subheader("Minimum balance per account")
    for name, m, d in acct_mins:
        st.write(f"**{name}** — {m:,.2f} on {d}")

    # Plot combined balance line chart
    st.subheader("Combined balance over time")
    try:
        st.line_chart(data=combined.set_index("date")[["balance","daily_change"]])
    except Exception:
        st.write("Chart failed to render.")

    # Calendar heatmap for combined balance (matplotlib)
    st.subheader("Calendar heatmap — combined daily balance")
    import matplotlib.pyplot as plt
    # build pivot: we will show one row per week of year; simpler approach: reshape into matrix by week number
    df_cal = combined.copy()
    df_cal["date"] = pd.to_datetime(df_cal["date"])
    df_cal["week"] = df_cal["date"].dt.isocalendar().week
    df_cal["dow"] = df_cal["date"].dt.weekday  # 0=Mon
    # create matrix weeks x 7 with NaN default
    weeks = sorted(df_cal["week"].unique())
    mat = np.full((len(weeks),7), np.nan)
    week_to_idx = {w:i for i,w in enumerate(weeks)}
    for _, row in df_cal.iterrows():
        wi = week_to_idx[int(row["week"])]
        di = int(row["dow"])
        mat[wi, di] = row["balance"]

    fig, ax = plt.subplots(figsize=(14, max(3, len(weeks)*0.25)))
    # Use imshow (no explicit colors specified), add colourbar
    im = ax.imshow(mat, aspect="auto", interpolation="nearest")
    ax.set_yticks(range(len(weeks)))
    ax.set_yticklabels([str(w) for w in weeks])
    ax.set_xticks(range(7))
    ax.set_xticklabels(["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])
    ax.set_title("Weekly rows — combined balance (empty = no data)")
    fig.colorbar(im, ax=ax, orientation="vertical", label="Balance (NZD)")
    st.pyplot(fig)

    # Show flagged days table
    flagged_combined = combined[combined["global_flag"] | combined["any_account_flag"]]
    if not flagged_combined.empty:
        st.warning(f"{len(flagged_combined)} day(s) flagged (global or per-account threshold).")
        display_cols = ["date","daily_change","balance","global_flag","any_account_flag"]
        st.dataframe(flagged_combined[display_cols].reset_index(drop=True))
    else:
        st.success("No flagged days in forecast window.")

    # Allow download of CSVs: combined + per-account
    csv_combined = combined.to_csv(index=False)
    st.download_button("Download combined forecast CSV", data=csv_combined, file_name="forecast_combined.csv", mime="text/csv")

    # Per-account exports
    for i, df in enumerate(per_acct_dfs):
        csv_acct = df.to_csv(index=False)
        st.download_button(f"Download {st.session_state['accounts'][i]['name']} forecast CSV", data=csv_acct, file_name=f"forecast_acct_{i}.csv", mime="text/csv")

except Exception as e:
    st.error(f"Forecast generation failed: {e}")

# Google Sheets integration (optional)
st.markdown("---")
st.header("Google Sheets integration (optional)")
if not GS_AVAILABLE:
    st.info("gspread or gspread_dataframe not installed or failed to import. To enable Google Sheets integration, install:\n\npip install gspread gspread_dataframe oauth2client")
else:
    creds = None
    gc = None
    if gs_json is not None:
        try:
            from google.oauth2.service_account import Credentials
            scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
            creds = Credentials.from_service_account_info(st.session_state.get("_gs_json_cache") or gs_json.getvalue(), scopes=scopes) if isinstance(st.session_state.get("_gs_json_cache"), dict) else None
        except Exception:
            # try reading file content
            try:
                import json
                sa_info = json.load(gs_json)
                from google.oauth2.service_account import Credentials
                scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
                creds = Credentials.from_service_account_info(sa_info, scopes=scopes)
            except Exception as ex:
                st.error(f"Failed to parse service account JSON: {ex}")

    if st.button("Test Google Sheets connection"):
        if creds is None:
            st.error("No valid service account JSON provided. Upload file or set credentials on host.")
        else:
            try:
                gc = gspread.authorize(creds)
                sh = None
                if gs_sheet_name:
                    try:
                        sh = gc.open(gs_sheet_name)
                        st.success(f"Connected to sheet: {gs_sheet_name}")
                    except Exception:
                        # try to create
                        try:
                            sh = gc.create(gs_sheet_name)
                            st.success(f"Created sheet: {gs_sheet_name}")
                        except Exception as ce:
                            st.error(f"Could not open or create sheet: {ce}")
                else:
                    st.info("Enter a Sheet name to open/create.")
            except Exception as ex:
                st.error(f"Google Sheets auth failed: {ex}")

    if st.button("Save forecast to Google Sheet (combined)"):
        if not GS_AVAILABLE:
            st.error("gspread not available.")
        else:
            if gs_json is None or not gs_sheet_name:
                st.error("Upload service account JSON and set a sheet name first.")
            else:
                try:
                    import json
                    sa_info = json.load(gs_json)
                    sc = Credentials.from_service_account_info(sa_info, scopes=["https://www.googleapis.com/auth/spreadsheets","https://www.googleapis.com/auth/drive"])
                    gc = gspread.authorize(sc)
                    sh = None
                    try:
                        sh = gc.open(gs_sheet_name)
                    except Exception:
                        sh = gc.create(gs_sheet_name)
                    # write combined
                    ws = sh.sheet1
                    df_to_write = combined.copy()
                    set_with_dataframe(ws, df_to_write)
                    st.success("Combined forecast saved to Google Sheet.")
                except Exception as ex:
                    st.error(f"Failed to save to Google Sheets: {ex}")

# Reset button (clear session)
st.markdown("---")
if st.button("Reset all (clear session state)"):
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.experimental_rerun()

st.caption("Built for NZD budgets. Contact app author for custom bank-business day rules or more persistence backends.")

# app.py
# Money Tracker & 1-year Forecast (minimal dependencies)
# Requirements: streamlit, pandas
# Run: streamlit run app.py

import streamlit as st
import pandas as pd
from datetime import date, timedelta
import calendar
import html as html_lib
import streamlit.components.v1 as components
import uuid

# ------------------------
# Frequency mapping
# ------------------------
FREQUENCIES = [
    "Once", "Daily", "Weekly", "Fortnightly", "4-weekly",
    "Monthly", "Quarterly", "6-monthly", "Annually"
]

# Map friendly label -> (unit, interval months/weeks/days)
FREQ_MAP = {
    "Once": ("once", 0),
    "Daily": ("days", 1),
    "Weekly": ("weeks", 1),
    "Fortnightly": ("weeks", 2),
    "4-weekly": ("weeks", 4),
    "Monthly": ("months", 1),
    "Quarterly": ("months", 3),
    "6-monthly": ("months", 6),
    "Annually": ("months", 12),
}

# ------------------------
# Helpers
# ------------------------
def ensure_key(k, default):
    if k not in st.session_state:
        st.session_state[k] = default

ensure_key("accounts", [])   # list of dicts: {id, name, opening, threshold}
ensure_key("items", [])      # list of dicts: {id, name, direction, amount, freq_label, start_date, end_date, shift_business, account_id, notes}
ensure_key("edit_item_index", None)
ensure_key("edit_account_index", None)

def safe_date_parse(d):
    if d is None or d == "":
        return None
    if isinstance(d, date):
        return d
    try:
        return pd.to_datetime(d).date()
    except Exception:
        return None

def normalize_amount(v):
    try:
        return float(v)
    except Exception:
        return None

def shift_to_business_day(d):
    d = safe_date_parse(d)
    if d is None:
        return None
    while d.weekday() >= 5:  # Sat=5 Sun=6
        d = d + timedelta(days=1)
    return d

def add_months(orig_date, months):
    y = orig_date.year + (orig_date.month - 1 + months) // 12
    m = (orig_date.month - 1 + months) % 12 + 1
    day = min(orig_date.day, calendar.monthrange(y, m)[1])
    return date(y, m, day)

def generate_occurrences(item, horizon_start, horizon_end):
    """
    Returns list of tuples (date, signed_amount, account_id)
    signed_amount positive for income, negative for expense
    """
    out = []
    amt = normalize_amount(item.get("amount"))
    if amt is None:
        return out
    signed = amt if item.get("direction") == "income" else -abs(amt)

    start = safe_date_parse(item.get("start_date"))
    if start is None:
        return out
    end = safe_date_parse(item.get("end_date")) or horizon_end

    unit, interval = FREQ_MAP.get(item.get("freq_label", "Once"), ("once", 0))
    shift = bool(item.get("shift_business", True))
    acct_id = item.get("account_id")

    # If series ends before horizon start, nothing to produce
    if end < horizon_start:
        return out

    current = start
    max_iter = 5000
    i = 0
    while i < max_iter and current <= end and current <= horizon_end:
        occ = shift_to_business_day(current) if shift else current
        if occ >= horizon_start and occ <= horizon_end:
            out.append((occ, signed, acct_id))

        # advance
        if unit == "once":
            break
        elif unit == "days":
            current = current + timedelta(days=interval)
        elif unit == "weeks":
            current = current + timedelta(weeks=interval)
        elif unit == "months":
            current = add_months(current, interval)
        else:
            break
        i += 1

    return out

def build_forecast(items, accounts_list, start_date, days_horizon, global_threshold=None):
    """
    Returns:
      cum_balances: DataFrame indexed by date with columns for each account + 'Total'
      daily_changes: DataFrame indexed by date with daily changes per account
      flags: dict with 'combined' Series and 'per_account' dict of Series
    """
    start_date = safe_date_parse(start_date) or date.today()
    horizon_end = start_date + timedelta(days=days_horizon - 1)
    date_index = pd.date_range(start_date, horizon_end, freq="D")

    if not accounts_list:
        return pd.DataFrame(), pd.DataFrame(), {}

    account_ids = [a['id'] for a in accounts_list]
    account_names = {a['id']: a['name'] for a in accounts_list}
    openings = {a['id']: float(a.get('opening', 0.0) or 0.0) for a in accounts_list}
    thresholds = {a['id']: float(a.get('threshold', float('nan'))) for a in accounts_list}

    daily_changes = pd.DataFrame(0.0, index=date_index, columns=[account_names[aid] for aid in account_ids])

    # accumulate events into daily_changes
    for it in items:
        occs = generate_occurrences(it, start_date, horizon_end)
        for d, amt, acct_id in occs:
            if acct_id not in account_ids:
                continue
            col = account_names[acct_id]
            try:
                ts = pd.to_datetime(d)
                if ts in daily_changes.index:
                    daily_changes.at[ts, col] += amt
            except Exception:
                continue

    # cumulative balances
    cum = daily_changes.cumsum()
    # add opening balances
    for aid in account_ids:
        col = account_names[aid]
        cum[col] = cum[col] + openings.get(aid, 0.0)

    cum["Total"] = cum.sum(axis=1)

    # flags
    flags = {}
    if global_threshold is not None:
        flags["combined"] = cum["Total"] < float(global_threshold)
    else:
        flags["combined"] = pd.Series([False]*len(cum), index=cum.index)

    per_account_flags = {}
    for aid in account_ids:
        col = account_names[aid]
        t = thresholds.get(aid, None)
        if pd.isna(t):
            per_account_flags[col] = pd.Series([False]*len(cum), index=cum.index)
        else:
            per_account_flags[col] = cum[col] < float(t)

    flags["per_account"] = per_account_flags

    return cum, daily_changes, flags

# color helpers for heatmap cells
def color_for_value(v, vmin, vmax):
    if v is None or pd.isna(v):
        return "#f5f5f5"
    try:
        v = float(v)
    except:
        return "#f5f5f5"
    if vmax == vmin:
        ratio = 0.5
    else:
        ratio = (v - vmin) / (vmax - vmin)
        ratio = max(0.0, min(1.0, ratio))
    r = int(255 * (1 - ratio))
    g = int(255 * ratio)
    b = 30
    return f"rgb({r},{g},{b})"

def render_calendar_heatmap_html(balances_df, account_name):
    """
    Builds a month-by-day calendar heatmap table as HTML for the provided account column.
    """
    if account_name not in balances_df.columns:
        return "<div>No data</div>"

    df = balances_df[[account_name]].reset_index().rename(columns={'index': 'date'})
    df['date'] = pd.to_datetime(df['date'])
    df['ym'] = df['date'].dt.strftime("%Y-%m")
    df['day'] = df['date'].dt.day

    months = df['ym'].unique().tolist()
    days = list(range(1, 32))
    vals = df[account_name].dropna()
    if vals.empty:
        vmin, vmax = 0.0, 0.0
    else:
        vmin, vmax = float(vals.min()), float(vals.max())

    mapping = {}
    for _, r in df.iterrows():
        mapping.setdefault(r['ym'], {})[int(r['day'])] = r[account_name]

    html = []
    html.append("<div style='font-family:Arial,Helvetica,sans-serif'>")
    html.append("<table style='border-collapse:collapse;width:100%'>")
    # header
    html.append("<thead><tr><th style='text-align:left;padding:6px;border-bottom:1px solid #ddd'>Month</th>")
    for d in days:
        html.append(f"<th style='text-align:center;padding:4px;border-bottom:1px solid #ddd'>{d}</th>")
    html.append("</tr></thead><tbody>")
    for ym in months:
        html.append(f"<tr><td style='padding:6px;border-bottom:1px solid #f0f0f0;background:#fafafa;font-weight:600'>{html_lib.escape(ym)}</td>")
        for d in days:
            v = mapping.get(ym, {}).get(d, None)
            if v is None or pd.isna(v):
                color = "#f5f5f5"
                disp = ""
            else:
                color = color_for_value(v, vmin, vmax)
                disp = f"{v:,.0f}"
            html.append(f"<td style='padding:6px;border-bottom:1px solid #f7f7f7;text-align:center;background:{color};min-width:44px'>{disp}</td>")
        html.append("</tr>")
    html.append("</tbody></table></div>")
    return "".join(html)


# ------------------------
# Streamlit UI
# ------------------------
st.set_page_config(page_title="Money Tracker & Forecast (NZD)", layout="wide")
st.title("💳 Money Tracker & Forecast — NZD")

# --- Accounts: add / edit / remove ---
st.header("Accounts")
st.write("Add accounts (any number). Each account has an opening balance and an optional low-balance threshold.")

a1, a2, a3 = st.columns([3,2,2])
with a1:
    new_acc_name = st.text_input("Account name", key="new_acc_name")
with a2:
    new_acc_open = st.number_input("Opening balance (NZD)", value=0.0, step=50.0, key="new_acc_open")
with a3:
    new_acc_thresh = st.number_input("Low threshold (optional)", value=0.0, step=50.0, key="new_acc_thresh")
if st.button("➕ Add account"):
    if not new_acc_name:
        st.error("Please enter an account name.")
    else:
        # ensure unique
        if any(a['name'] == new_acc_name for a in st.session_state['accounts']):
            st.error("An account with that name already exists.")
        else:
            st.session_state['accounts'].append({
                "id": str(uuid.uuid4()),
                "name": new_acc_name,
                "opening": float(new_acc_open),
                "threshold": float(new_acc_thresh)
            })
            st.success(f"Added account: {new_acc_name}")
            st.experimental_rerun()

# Edit or remove
if st.session_state['accounts']:
    st.write("**Existing accounts**")
    df_accounts = pd.DataFrame(st.session_state['accounts']).loc[:, ["name", "opening", "threshold"]].rename(columns={"opening": "Opening (NZD)", "threshold": "Low threshold"})
    st.dataframe(df_accounts)

    sel_acc_name = st.selectbox("Select account to edit / remove", options=[a['name'] for a in st.session_state['accounts']], key="sel_acc")
    sel_idx = next((i for i,a in enumerate(st.session_state['accounts']) if a['name'] == sel_acc_name), None)
    if sel_idx is not None:
        acc = st.session_state['accounts'][sel_idx]
        col_e1, col_e2, col_e3 = st.columns([3,2,2])
        with col_e1:
            edit_name = st.text_input("Name", value=acc['name'], key=f"edit_name_{acc['id']}")
        with col_e2:
            edit_open = st.number_input("Opening (NZD)", value=float(acc.get("opening",0.0)), key=f"edit_open_{acc['id']}")
        with col_e3:
            edit_thresh = st.number_input("Low threshold", value=float(acc.get("threshold",0.0)), key=f"edit_thresh_{acc['id']}")
        if st.button("Update account", key=f"update_acc_{acc['id']}"):
            # update account and also update any items referencing old name (account_id stays same)
            st.session_state['accounts'][sel_idx]['name'] = edit_name
            st.session_state['accounts'][sel_idx]['opening'] = float(edit_open)
            st.session_state['accounts'][sel_idx]['threshold'] = float(edit_thresh)
            st.success("Account updated.")
            st.experimental_rerun()

        # remove account handling with options for items assigned
        # count items assigned
        assigned_items = [it for it in st.session_state['items'] if it.get("account_id") == acc['id']]
        if assigned_items:
            st.warning(f"{len(assigned_items)} planned item(s) are assigned to this account.")
            rem_choice = st.selectbox("When removing this account, do you want to:", options=["Cancel","Delete those items","Reassign to another account"], key=f"rem_choice_{acc['id']}")
            if rem_choice == "Reassign to another account":
                other_accounts = [a for a in st.session_state['accounts'] if a['id'] != acc['id']]
                if other_accounts:
                    reassign_to = st.selectbox("Reassign items to:", options=[a['name'] for a in other_accounts], key=f"reassign_to_{acc['id']}")
                    if st.button("🗑️ Remove account and reassign items", key=f"remove_reassign_{acc['id']}"):
                        # find target id
                        target_id = next(a['id'] for a in other_accounts if a['name'] == reassign_to)
                        # reassign
                        for it in st.session_state['items']:
                            if it.get("account_id") == acc['id']:
                                it['account_id'] = target_id
                        st.session_state['accounts'].pop(sel_idx)
                        st.success("Account removed and items reassigned.")
                        st.experimental_rerun()
                else:
                    st.info("No other accounts to reassign to. Create another account first.")
            elif rem_choice == "Delete those items":
                if st.button("🗑️ Remove account and delete items", key=f"remove_del_{acc['id']}"):
                    # remove items assigned
                    st.session_state['items'] = [it for it in st.session_state['items'] if it.get("account_id") != acc['id']]
                    st.session_state['accounts'].pop(sel_idx)
                    st.success("Account and its items removed.")
                    st.experimental_rerun()
        else:
            if st.button("🗑️ Remove account (no items assigned)", key=f"remove_acc_{acc['id']}"):
                st.session_state['accounts'].pop(sel_idx)
                st.success("Account removed.")
                st.experimental_rerun()

else:
    st.info("No accounts yet. Add an account above to start planning.")

st.markdown("---")

# --- Planned items (add / edit / remove) ---
st.header("Planned incomes & expenses")
st.write("Add one-off or recurring planned incomes/expenses and assign them to an account. Business-day shift (weekend -> next Monday) is enabled by default.")

# Prepare account selection options
acc_options = {a['id']: a['name'] for a in st.session_state['accounts']}

with st.form("item_form", clear_on_submit=False):
    left, mid, right = st.columns([3,2,2])
    with left:
        item_name = st.text_input("Name", key="item_name")
        direction = st.selectbox("Type", options=["income","expense"], index=0, key="item_direction")
        notes = st.text_area("Notes (optional)", height=80, key="item_notes")
    with mid:
        amount = st.number_input("Amount (NZD)", value=0.0, step=10.0, key="item_amount")
        freq_label = st.selectbox("Frequency", options=FREQUENCIES, index=FREQUENCIES.index("Monthly"), key="item_freq")
        start_date = st.date_input("Start date", value=date.today(), key="item_start")
        end_date = st.date_input("End date (optional)", value=None, key="item_end")
    with right:
        shift_business = st.checkbox("Shift weekend -> next Monday", value=True, key="item_shift")
        if acc_options:
            account_choice = st.selectbox("Account", options=list(acc_options.values()), key="item_account")
        else:
            account_choice = None
        submit_item = st.form_submit_button("➕ Add / Update item")

    if submit_item:
        errors = []
        if not item_name:
            errors.append("Item name required.")
        if normalize_amount(amount) is None:
            errors.append("Amount must be a number.")
        if account_choice is None:
            errors.append("Please create at least one account and assign the item.")
        if safe_date_parse(start_date) is None:
            errors.append("Invalid start date.")
        if end_date and safe_date_parse(end_date) and safe_date_parse(end_date) < safe_date_parse(start_date):
            errors.append("End date cannot be before start date.")
        if errors:
            for e in errors:
                st.error(e)
        else:
            # find account id
            acct_id = None
            for aid, name in acc_options.items():
                if name == account_choice:
                    acct_id = aid
                    break
            if st.session_state.get("edit_item_index") is None:
                # add new
                st.session_state['items'].append({
                    "id": str(uuid.uuid4()),
                    "name": item_name,
                    "direction": direction,
                    "amount": float(amount),
                    "freq_label": freq_label,
                    "start_date": safe_date_parse(start_date),
                    "end_date": safe_date_parse(end_date),
                    "shift_business": bool(shift_business),
                    "account_id": acct_id,
                    "notes": notes
                })
                st.success("Item added.")
            else:
                # update existing
                ei = st.session_state["edit_item_index"]
                if 0 <= ei < len(st.session_state["items"]):
                    st.session_state['items'][ei].update({
                        "name": item_name,
                        "direction": direction,
                        "amount": float(amount),
                        "freq_label": freq_label,
                        "start_date": safe_date_parse(start_date),
                        "end_date": safe_date_parse(end_date),
                        "shift_business": bool(shift_business),
                        "account_id": acct_id,
                        "notes": notes
                    })
                    st.success("Item updated.")
                    st.session_state["edit_item_index"] = None
            st.experimental_rerun()

# Items list
st.markdown("---")
st.write("Current planned items")
if st.session_state['items']:
    # Prepare table for display: show account names instead of ids
    display_rows = []
    for i, it in enumerate(st.session_state['items']):
        acct_name = next((a['name'] for a in st.session_state['accounts'] if a['id'] == it.get('account_id')), "(no account)")
        display_rows.append({
            "index": i,
            "name": it.get("name"),
            "type": it.get("direction"),
            "amount": it.get("amount"),
            "frequency": it.get("freq_label"),
            "start": it.get("start_date"),
            "end": it.get("end_date"),
            "account": acct_name,
            "notes": it.get("notes")
        })
    st.dataframe(pd.DataFrame(display_rows).loc[:, ["index","name","type","amount","frequency","start","end","account","notes"]])

    col_a, col_b = st.columns(2)
    with col_a:
        load_idx = st.number_input("Index to load for edit", min_value=0, max_value=len(st.session_state['items'])-1, value=0, key="load_idx")
        if st.button("Load item for edit"):
            it = st.session_state['items'][int(load_idx)]
            # populate the form fields
            st.session_state['item_name'] = it.get("name")
            st.session_state['item_direction'] = it.get("direction")
            st.session_state['item_amount'] = float(it.get("amount") or 0.0)
            st.session_state['item_freq'] = it.get("freq_label")
            st.session_state['item_start'] = it.get("start_date")
            st.session_state['item_end'] = it.get("end_date")
            st.session_state['item_shift'] = bool(it.get("shift_business", True))
            acct_name = next((a['name'] for a in st.session_state['accounts'] if a['id'] == it.get("account_id")), None)
            if acct_name:
                st.session_state['item_account'] = acct_name
            st.session_state["edit_item_index"] = int(load_idx)
            st.experimental_rerun()
    with col_b:
        rem_idx = st.number_input("Index to remove", min_value=0, max_value=len(st.session_state['items'])-1, value=0, key="rem_idx")
        if st.button("Remove item at index"):
            removed = st.session_state['items'].pop(int(rem_idx))
            st.success(f"Removed item: {removed.get('name')}")
            st.experimental_rerun()
else:
    st.info("No planned items yet. Add one above.")

st.markdown("---")

# --- Forecast ---
st.header("Forecast")
f1, f2, f3 = st.columns([2,2,2])
with f1:
    forecast_start = st.date_input("Forecast start date", value=date.today())
with f2:
    days = st.number_input("Forecast days", min_value=30, max_value=730, value=365)
with f3:
    global_threshold = st.number_input("Global low balance threshold (Total NZD)", value=0.0, step=50.0)

if st.button("Generate forecast"):
    try:
        cum_balances, daily_changes, flags = build_forecast(st.session_state['items'], st.session_state['accounts'], forecast_start, int(days), global_threshold)
        if cum_balances.empty:
            st.error("Add at least one account to generate a forecast.")
        else:
            st.success("Forecast generated.")
            st.subheader("Combined & per-account balances (first 180 rows)")
            st.dataframe(cum_balances.head(180))

            min_total = cum_balances["Total"].min()
            min_date = cum_balances["Total"].idxmin()
            st.metric("Minimum combined balance", f"${min_total:,.2f}", delta=None)

            flagged_total_days = int(flags["combined"].sum()) if "combined" in flags else 0
            if flagged_total_days > 0:
                st.warning(f"{flagged_total_days} day(s) where combined balance < global threshold (${global_threshold:,.2f})")
            else:
                st.success("Combined balance stays above the global threshold in the forecast window.")

            st.subheader("Per-account minimums")
            mins = []
            for a in st.session_state['accounts']:
                name = a['name']
                if name in cum_balances.columns:
                    mn = cum_balances[name].min()
                    md = cum_balances[name].idxmin()
                    mins.append({"Account": name, "Min balance (NZD)": f"${mn:,.2f}", "Date": md})
            st.table(pd.DataFrame(mins))

            st.subheader("Flagged days (combined OR per-account thresholds)")
            any_acct_flags = pd.Series(False, index=cum_balances.index)
            for colname, series in flags.get("per_account", {}).items():
                any_acct_flags = any_acct_flags | series
            flagged_any = flags.get("combined", pd.Series([False]*len(cum_balances), index=cum_balances.index)) | any_acct_flags
            if flagged_any.any():
                st.dataframe(cum_balances[flagged_any])
            else:
                st.write("No flagged days in forecast window.")

            st.subheader("Calendar heatmaps (per-account)")
            for a in st.session_state['accounts']:
                name = a['name']
                st.markdown(f"**{name}**")
                html = render_calendar_heatmap_html(cum_balances, name)
                # estimate height: ~ 40px per month row
                month_count = len(pd.to_datetime(cum_balances.index).to_period("M").unique())
                height = 120 + (month_count * 28)
                components.html(html, height=height, scrolling=True)
                # CSV download
                csv = cum_balances[[name]].to_csv()
                st.download_button(f"Download {name} CSV", data=csv, file_name=f"forecast_{name}.csv", mime="text/csv")

            # combined CSV
            csv_comb = cum_balances.to_csv()
            st.download_button("Download combined forecast CSV", data=csv_comb, file_name="forecast_combined.csv", mime="text/csv")

    except Exception as e:
        st.error(f"Forecast generation failed: {e}")

st.markdown("---")
if st.button("Reset everything (clear session)"):
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.experimental_rerun()

st.caption("Built for NZD budgets. If you'd like Google Sheets persistence or an alternate calendar style, I can add it next (that requires extra packages).")

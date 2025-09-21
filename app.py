
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import io

st.set_page_config(page_title="Money Tracker", layout="wide")

def parse_date(s):
    if pd.isna(s) or s == "":
        return None
    return pd.to_datetime(s).date()

if "items" not in st.session_state:
    # default example items
    st.session_state.items = [
        {"Name":"Salary","Type":"Income","Amount":2000,"Start Date":"2025-10-01","Frequency":"Fortnightly","Notes":"Pay"},
        {"Name":"Mortgage","Type":"Expense","Amount":1650,"Start Date":"2025-10-02","Frequency":"Fortnightly","Notes":"House2"},
        {"Name":"Food","Type":"Expense","Amount":150,"Start Date":"2025-10-03","Frequency":"Weekly","Notes":"Groceries"},
        {"Name":"Utilities","Type":"Expense","Amount":300,"Start Date":"2025-10-05","Frequency":"Monthly","Notes":"Power/Internet"}
    ]

st.title("Money Tracker — Web App")
st.markdown("Enter planned incomes and expenses and get a daily forecast of your expected balance. All amounts are monetary (NZD recommended).")

with st.sidebar:
    st.header("Add / Import Items")
    with st.form("add_item", clear_on_submit=True):
        name = st.text_input("Name", value="New item")
        typ = st.selectbox("Type", ["Income","Expense"])
        amount = st.number_input("Amount", min_value=0.0, value=100.0, step=1.0, format="%.2f")
        start_date = st.date_input("Start Date", value=datetime.today().date())
        freq = st.selectbox("Frequency", ["Daily","Weekly","Fortnightly","Monthly","Once"])
        notes = st.text_input("Notes (optional)","")
        submitted = st.form_submit_button("Add item")
        if submitted:
            st.session_state.items.append({"Name":name,"Type":typ,"Amount":float(amount),"Start Date":start_date.isoformat(),"Frequency":freq,"Notes":notes})
            st.success("Item added")

    st.markdown("---")
    uploaded = st.file_uploader("Import CSV of items (columns: Name,Type,Amount,Start Date,Frequency,Notes)", type=["csv"])
    if uploaded is not None:
        try:
            df_up = pd.read_csv(uploaded)
            for _, r in df_up.iterrows():
                st.session_state.items.append({
                    "Name": r.get("Name","Imported"),
                    "Type": r.get("Type","Expense"),
                    "Amount": float(r.get("Amount",0)),
                    "Start Date": pd.to_datetime(r.get("Start Date")).date().isoformat() if not pd.isna(r.get("Start Date")) else datetime.today().date().isoformat(),
                    "Frequency": r.get("Frequency","Monthly"),
                    "Notes": r.get("Notes","")
                })
            st.success("Imported CSV items")
        except Exception as e:
            st.error(f"Failed to import CSV: {e}")

    st.markdown("---")
    st.header("Forecast settings")
    col1, col2 = st.columns(2)
    with col1:
        start = st.date_input("Forecast start date", value=datetime.today().date())
    with col2:
        end = st.date_input("Forecast end date", value=(datetime.today().date() + timedelta(days=90)))
    opening_balance = st.number_input("Opening balance (NZD)", value=0.0, step=1.0, format="%.2f")
    st.markdown("---")
    if st.button("Clear all items"):
        st.session_state.items = []
        st.experimental_rerun()

# Main area: show items and allow editing/deleting
st.subheader("Planned incomes & expenses")
df_items = pd.DataFrame(st.session_state.items)
if df_items.empty:
    st.info("No items yet — add them in the sidebar")
else:
    # provide simple edit/delete by index
    st.dataframe(df_items, use_container_width=True)

    st.markdown("""**Delete an item**: enter the index (left-most row number starting at 0) and press Delete.""")
    del_idx = st.number_input("Index to delete", min_value=0, value=0, step=1)
    if st.button("Delete"):
        if 0 <= del_idx < len(st.session_state.items):
            st.session_state.items.pop(int(del_idx))
            st.success("Deleted")
            st.experimental_rerun()
        else:
            st.error("Index out of range")

# Compute daily forecast
def occurs_on(day, start_dt, freq):
    if day < start_dt:
        return False
    if freq == "Daily":
        return True
    if freq == "Weekly":
        return (day - start_dt).days % 7 == 0
    if freq == "Fortnightly":
        return (day - start_dt).days % 14 == 0
    if freq == "Monthly":
        # occur on same day-of-month; if not available (e.g. 31st), skip that month
        return day.day == start_dt.day
    if freq == "Once":
        return day == start_dt
    return False

items = []
for it in st.session_state.items:
    try:
        items.append({"Name":it["Name"], "Type":it["Type"], "Amount":float(it["Amount"]), "Start": parse_date(it["Start Date"]), "Frequency":it["Frequency"], "Notes":it.get("Notes","")})
    except Exception:
        # skip invalid rows
        pass

dates = pd.date_range(start, end, freq='D').date
rows = []
balance = float(opening_balance)
for d in dates:
    income = 0.0
    expense = 0.0
    for it in items:
        sd = it["Start"]
        if sd is None:
            continue
        if occurs_on(d, sd, it["Frequency"]):
            if it["Type"] == "Income":
                income += it["Amount"]
            else:
                expense += it["Amount"]
    net = income - expense
    balance += net
    rows.append({"Date":d, "Income":income, "Expenses":expense, "Net Change":net, "Balance":balance})

df_forecast = pd.DataFrame(rows)
st.subheader("Daily Forecast")
st.dataframe(df_forecast, use_container_width=True)

# Plot
st.subheader("Balance over time")
st.line_chart(df_forecast.set_index("Date")["Balance"])

# Downloads
def to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Forecast')
        pd.DataFrame(st.session_state.items).to_excel(writer, index=False, sheet_name='Inputs')
    processed_data = output.getvalue()
    return processed_data

st.download_button("Download forecast CSV", df_forecast.to_csv(index=False).encode('utf-8'), file_name='forecast.csv', mime='text/csv')

st.download_button("Download forecast & inputs Excel", to_excel(df_forecast), file_name='money_tracker_forecast.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

st.markdown("""---
**Notes & tips**:
- Frequencies supported: Daily, Weekly (every 7 days), Fortnightly (every 14 days), Monthly (same day number), Once.
- If a monthly start date is the 29/30/31 and a month lacks that day the event will skip that month.
- Open-source: you can host this on Streamlit Cloud or run locally with `streamlit run app.py`.
""")

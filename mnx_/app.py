import streamlit as st
import pandas as pd
import plotly.express as px
from fpdf import FPDF
import warnings
import tempfile
import os

# Suppress warnings
warnings.filterwarnings("ignore")

# Optional forecasting import
try:
    from sklearn.linear_model import LinearRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# 1. Page config
st.set_page_config(
    page_title="Zimbabwe Crime Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Main title
st.title("CRIME PREDICTION MODEL")

# 2. Data load & upload function
def upload_csv() -> pd.DataFrame | None:
    """
    Show a sidebar uploader for CSV files. If a file is uploaded,
    read and return it as a DataFrame; otherwise return None.
    """
    uploaded_file = st.sidebar.file_uploader(
        label="Upload crime data CSV", type=["csv"]
    )
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, parse_dates=["Date"])
            st.sidebar.success(f"Loaded {len(df)} records from CSV")
            return df
        except Exception as e:
            st.sidebar.error(f"Error loading CSV: {e}")
    return None

@st.cache_data
# Fallback loader for default file
def load_data(path: str = "crime_data.csv") -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["Date"])

# 2b. Load DataFrame: uploaded or default
uploaded_df = upload_csv()
if uploaded_df is not None:
    df = uploaded_df
else:
    df = load_data()

# 3. Filters
st.sidebar.header("🔎 Filters")
min_date, max_date = df["Date"].min().date(), df["Date"].max().date()
start = st.sidebar.date_input("Start date", min_date, min_date, max_date)
end   = st.sidebar.date_input("End date",   max_date, min_date, max_date)
category = st.sidebar.selectbox(
    "Crime category", ["All"] + list(df["Crime_Type"].unique())
)
districts = st.sidebar.multiselect(
    "Districts", df["District"].unique(), df["District"].unique()
)
mask = (
    df["Date"].dt.date.between(start, end)
    & df["District"].isin(districts)
)
if category != "All":
    mask &= df["Crime_Type"] == category
filtered = df[mask]

# 4. Metrics
tot = len(filtered)
ucat = filtered["Crime_Type"].nunique()
dud = filtered["District"].nunique()
c1, c2, c3 = st.columns(3)
c1.metric("Total Incidents", tot)
c2.metric("Crime Categories", ucat)
c3.metric("Affected Districts", dud)
st.markdown("---")

# 5. Time series
fig_ts = px.line(
    filtered.groupby(pd.Grouper(key="Date", freq="ME")).size().reset_index(name="Count"),
    x="Date", y="Count", title="Incidents Over Time"
)
st.plotly_chart(fig_ts, use_container_width=True)

# 6. Pie chart
fig_pie = px.pie(
    filtered["Crime_Type"].value_counts().reset_index(name="Count").rename(columns={"index":"Crime_Type"}),
    names="Crime_Type", values="Count", title="Crime Type Breakdown", hole=0.4
)
st.plotly_chart(fig_pie, use_container_width=True)

# 7. Bar chart
fig_bar = px.bar(
    filtered.groupby("District").size().reset_index(name="Count"),
    x="District", y="Count", title="Incidents per District"
)
st.plotly_chart(fig_bar, use_container_width=True)

# 8. Map
fig_map = px.scatter_mapbox(
    filtered, lat="Latitude", lon="Longitude", color="Crime_Type",
    hover_data=["Date","District"], zoom=6, height=400
)
fig_map.update_layout(mapbox_style="open-street-map", margin={"r":0,"t":0,"l":0,"b":0})
st.plotly_chart(fig_map, use_container_width=True)

# 9. PDF export with images (requires kaleido: pip install kaleido)
def generate_pdf(metrics: dict, figs: list) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(True, margin=15)
    # Title page
    pdf.add_page()
    pdf.set_font("Arial","B",16)
    pdf.cell(0,10,"Zimbabwe Crime Dashboard Report",ln=True,align="C")
    pdf.ln(5)
    pdf.set_font("Arial",size=12)
    pdf.cell(0,8,f"Total incidents: {metrics['total']}",ln=True)
    pdf.cell(0,8,f"Crime categories: {metrics['categories']}",ln=True)
    pdf.cell(0,8,f"Affected districts: {metrics['districts']}",ln=True)
    # Insert figures
    for fig in figs:
        tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        try:
            fig.write_image(tmp.name, format='png')
            pdf.add_page()
            pdf.image(tmp.name, x=10, y=20, w=pdf.w-20)
        except Exception as e:
            st.error(f"Error embedding figure: {e}")
        finally:
            tmp.close()
            os.remove(tmp.name)
    return pdf.output(dest='S').encode('latin-1')

if st.sidebar.button("Export Dashboard as PDF"):
    metrics = {"total": tot, "categories": ucat, "districts": dud}
    figs = [fig_ts, fig_pie, fig_bar, fig_map]
    pdf_bytes = generate_pdf(metrics, figs)
    st.sidebar.download_button("Download PDF", pdf_bytes, "crime_dashboard.pdf", "application/pdf")

# 10. Optional forecasting
if SKLEARN_AVAILABLE:
    st.sidebar.header("🔮 Forecast")
    pdist = st.sidebar.selectbox("District for forecast", df["District"].unique())
    hist = (df[df["District"]==pdist].groupby(pd.Grouper(key="Date",freq="ME")).size().reset_index(name="Count"))
    if len(hist)>1:
        hist["Mnum"] = hist["Date"].dt.year*12+hist["Date"].dt.month
        model = LinearRegression().fit(hist[["Mnum"]], hist["Count"])
        nm = df["Date"].max().year*12+df["Date"].max().month+1
        pred = model.predict([[nm]])[0]
        st.sidebar.metric(f"Next-month {pdist}",f"{pred:.0f}")
else:
    st.sidebar.warning("Install scikit-learn for forecasting.")

import json
import logging
import sys
from hashlib import sha256
from pathlib import Path

import pandas as pd
import streamlit as st

# Streamlit Cloud executes this file from the dashboard directory.
# Add the repository root so shared helpers in src/ are importable.
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.upload_health import analyze_uploaded_csv, create_spark_session
from src.mongo_store import load_recent_reports, save_report
from src.ml_anomaly import run_behavioral_anomaly_model

LOGGER = logging.getLogger(__name__)

st.set_page_config(page_title="Intelligent Big Data Monitoring", page_icon="📊", layout="wide")

MONTHLY_FILE = BASE_DIR / "fingerprints" / "monthly_fingerprints.json"
BEHAVIORAL_FILE = BASE_DIR / "fingerprints" / "behavioral_drift.json"
ALERT_FILE = BASE_DIR / "fingerprints" / "alert_history.json"

FEATURE_LABELS = {
    "purchase_record_count": "Purchase Record Count",
    "survey_response_diversity": "Survey Response Diversity",
    "unique_product_code_count": "Unique Product Code Count",
    "unique_category_count": "Unique Category Count",
    "unique_shipping_state_count": "Unique Shipping State Count",
    "average_quantity": "Average Quantity",
    "quantity_stddev": "Quantity Standard Deviation",
    "average_purchase_price_per_unit": "Average Purchase Price Per Unit",
    "purchase_price_per_unit_stddev": "Purchase Price Per Unit Standard Deviation",
    "missing_value_rate": "Missing-Value Rate",
    "duplicate_rate": "Duplicate-Row Rate",
}
SEVERITY_RANK = {"INFO / NORMAL": 0, "WARNING": 1, "HIGH": 2, "CRITICAL": 3}


@st.cache_data
def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def show_severity(message, severity):
    """Use native Streamlit status components while retaining visible severity text."""
    visible_message = f"**{severity}** — {message}"
    if severity == "INFO / NORMAL":
        st.success(visible_message)
    elif severity == "WARNING":
        st.warning(visible_message)
    else:  # HIGH and CRITICAL are immediately actionable statuses.
        st.error(visible_message)


def short_reason(reason, limit=105):
    if not reason:
        return "No reason recorded"
    return reason if len(reason) <= limit else f"{reason[:limit - 1].rstrip()}…"


def get_mongodb_settings():
    """Read MongoDB configuration from Streamlit secrets without exposing it."""
    try:
        uri = st.secrets["MONGODB_URI"]
    except (KeyError, FileNotFoundError):
        return None
    return {
        "uri": uri,
        "database": st.secrets.get("MONGODB_DATABASE", "bda_monitoring"),
        "collection": st.secrets.get("MONGODB_COLLECTION", "dataset_health_reports"),
    }


@st.cache_resource
def get_spark_session():
    return create_spark_session()


monthly = load_json(str(MONTHLY_FILE))
behavioral = load_json(str(BEHAVIORAL_FILE))
alerts = load_json(str(ALERT_FILE)) if ALERT_FILE.exists() else []
mongodb_settings = get_mongodb_settings()

st.sidebar.title("Data Health Monitor")
st.sidebar.subheader("Active Dataset")
st.sidebar.write("Open e-commerce 1.0 — Amazon Purchases")
st.sidebar.caption("Genuine 2022 purchase records only")
st.sidebar.divider()
st.sidebar.write("Apache Spark / PySpark")
st.sidebar.write("Parquet")
st.sidebar.write("Spark MLlib")
st.sidebar.write("MongoDB Atlas / NoSQL")
st.sidebar.write("Streamlit")

st.title("Intelligent Big Data Quality & Behavioral Drift Monitoring System using PySpark")
st.write("PySpark monitoring of genuine 2022 purchase records from the public Open e-commerce 1.0 Amazon Purchases dataset.")
st.info(
    "No transaction dates were altered and no synthetic 2023/2024 data was created. "
    "Survey ResponseID is shown only as survey-response diversity; it is not treated as a confirmed customer identifier."
)

st.divider()
st.header("Analyze Your Dataset")
st.caption("Upload your CSV file to perform an automated data health assessment.")
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if not uploaded_file:
    st.info("Upload a .csv file to see schema, missing values, duplicates, outliers, and a prototype data health score.")
else:
    st.write(f"**File Name:** {uploaded_file.name}")
    try:
        spark = get_spark_session()
        analysis, error = analyze_uploaded_csv(spark, uploaded_file)
    except Exception as exc:
        LOGGER.exception("PySpark startup failed for uploaded CSV analysis")
        analysis = None
        error = (
            "Unable to start PySpark for this upload. "
            f"Details: {type(exc).__name__}: {exc}"
        )
    if error:
        st.error(error)
    elif analysis is None:
        st.error("Unable to analyze the uploaded file.")
    else:
        st.write(f"**Rows:** {analysis.row_count:,}")
        st.write(f"**Columns:** {analysis.column_count}")
        left, middle, right, right2, right3, right4 = st.columns(6)
        left.metric("Data Health Score", f"{analysis.data_health_score}/100")
        middle.metric("Health Status", analysis.health_status)
        right.metric("Rows", f"{analysis.row_count:,}")
        right2.metric("Columns", analysis.column_count)
        right3.metric("Missing %", f"{analysis.overall_missing_percentage:.2f}%")
        right4.metric("Duplicate %", f"{analysis.duplicate_percentage:.2f}%")

        if analysis.date_detection_message:
            st.info(analysis.date_detection_message)

        st.caption("Prototype configurable health thresholds")

        if not analysis.has_valid_rows:
            st.warning("The uploaded CSV appears to be empty or unreadable enough to analyze safely.")

        if not analysis.has_readable_columns:
            st.warning("No readable columns were detected.")

        st.subheader("Schema Overview")
        st.dataframe(pd.DataFrame(analysis.schema_overview), use_container_width=True, hide_index=True)

        st.subheader("Missing Values by Column")
        st.dataframe(pd.DataFrame(analysis.missing_by_column), use_container_width=True, hide_index=True, column_config={"Missing %": st.column_config.NumberColumn(format="%.2f%%")})

        st.subheader("Numerical Summary")
        if analysis.numerical_summary:
            st.dataframe(pd.DataFrame(analysis.numerical_summary), use_container_width=True, hide_index=True, column_config={
                "Mean": st.column_config.NumberColumn(format="%.4f"),
                "Std Dev": st.column_config.NumberColumn(format="%.4f"),
                "Minimum": st.column_config.NumberColumn(format="%.4f"),
                "Maximum": st.column_config.NumberColumn(format="%.4f"),
            })
        else:
            st.info("No numerical columns were detected.")

        st.subheader("Outlier Analysis")
        if analysis.outlier_analysis:
            st.dataframe(pd.DataFrame(analysis.outlier_analysis), use_container_width=True, hide_index=True, column_config={"Outlier %": st.column_config.NumberColumn(format="%.2f%%")})
        else:
            st.info("No suitable numerical columns were found for IQR outlier analysis.")

        st.subheader("Detected Issues")
        for issue in analysis.detected_issues:
            st.write(f"- {issue}")

        with st.expander("How is the Data Health Score calculated?"):
            st.write("Score starts at 100 and is reduced by transparent penalties.")
            st.markdown(
                """
                - Missing-value penalty:
                  - <= 1%: 0
                  - > 1% and <= 5%: -10
                  - > 5% and <= 15%: -20
                  - > 15%: -30
                - Duplicate penalty:
                  - <= 1%: 0
                  - > 1% and <= 5%: -10
                  - > 5% and <= 15%: -20
                  - > 15%: -30
                - Outlier penalty:
                  - <= 2%: 0
                  - > 2% and <= 5%: -5
                  - > 5% and <= 10%: -10
                  - > 10%: -20
                - Optional usability penalty:
                  - Applied only when the dataset has serious usability problems such as unreadable columns or no valid rows.
                """
            )
            st.caption("These are prototype configurable health thresholds, not universal standards.")

        if mongodb_settings:
            report_key = sha256(uploaded_file.getvalue()).hexdigest()
            if st.button("Save this report to MongoDB Atlas", key=f"save_mongo_{report_key}"):
                try:
                    with st.spinner("Saving report to MongoDB Atlas..."):
                        save_report(
                            mongodb_settings["uri"],
                            mongodb_settings["database"],
                            mongodb_settings["collection"],
                            analysis,
                        )
                    st.success("Report saved to MongoDB Atlas.")
                except Exception:
                    st.error("MongoDB could not save this report. Check the Atlas URI, database user, and network access.")
        else:
            st.caption("MongoDB storage is available after MONGODB_URI is added to Streamlit secrets.")

total_records = sum(item["total_records"] for item in monthly)
overview_records, overview_dates, overview_weeks = st.columns([1.15, 1.75, 1.1])
overview_records.metric("2022 Purchase Records", f"{total_records:,}")
overview_dates.metric("Active Date Range", "2022-01-01 to 2022-12-31")
overview_weeks.metric("Weekly Batches", len(behavioral))

st.divider()
st.header("IMPLEMENTATION 1 — Data Health Profiling")
st.caption("Is the current data batch technically healthy?")
months = [item["month"] for item in monthly]
selected_month = st.selectbox("Select Purchase Month", months, index=len(months) - 1)
month_data = next(item for item in monthly if item["month"] == selected_month)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Health Score", f"{month_data['health_score']}/100")
c2.metric("Health Status", month_data["health_status"])
c3.metric("Purchase Records", f"{month_data['total_records']:,}")
c4.metric("Columns", month_data["total_columns"])

monthly_trend = pd.DataFrame(
    [{"Month": item["month"], "Health Score": item["health_score"], "Purchase Record Count": item["total_records"]} for item in monthly]
).set_index("Month")
health_chart, volume_chart = st.columns(2)
with health_chart:
    st.subheader("Monthly Data Health Score Trend")
    st.line_chart(monthly_trend[["Health Score"]], height=300)
with volume_chart:
    st.subheader("Monthly Purchase Record Volume")
    st.line_chart(monthly_trend[["Purchase Record Count"]], height=300)

st.subheader("Missing Values by Amazon Field")
st.dataframe(
    pd.DataFrame(
        [
            {"Column": column, "Missing Count": month_data["missing_values_per_column"][column], "Missing %": value}
            for column, value in month_data["missing_percentage_per_column"].items()
        ]
    ),
    use_container_width=True,
    hide_index=True,
    column_config={"Missing %": st.column_config.NumberColumn(format="%.2f%%")},
)
st.subheader("Numerical Summary")
st.dataframe(
    pd.DataFrame(
        [
            {
                "Field": field,
                "Count": values["count"],
                "Mean": values["mean"],
                "Std Dev": values["stddev"],
                "Minimum": values["min"],
                "Maximum": values["max"],
            }
            for field, values in month_data["numerical_statistics"].items()
        ]
    ),
    use_container_width=True,
    hide_index=True,
)
with st.expander("View IQR Outlier Analysis and Schema Monitoring"):
    st.dataframe(
        pd.DataFrame(
            [
                {"Field": field, "Q1": values["q1"], "Q3": values["q3"], "Outlier %": values["outlier_percentage"]}
                for field, values in month_data["outlier_analysis"].items()
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.write("No schema drift detected." if not month_data["schema_drift"]["detected"] else "Schema drift detected.")
    st.write("Configured anomalies:", month_data["data_quality_anomalies"] or "None")

st.divider()
st.header("IMPLEMENTATION 2 — Behavioral Drift Intelligence")
st.caption("Is the data behaving normally?")
weeks = [item["week"] for item in behavioral]
default_week = "2022-07-11" if "2022-07-11" in weeks else weeks[-1]
selected_week = st.selectbox("Select Weekly Purchase Batch", weeks, index=weeks.index(default_week))
week_data = next(item for item in behavioral if item["week"] == selected_week)
b1, b2, b3, b4 = st.columns(4)
b1.metric("Selected Week", week_data["week"])
b2.metric(
    "Behavioral Similarity Score",
    f"{week_data['behavioral_similarity_score']}/100" if week_data["behavioral_similarity_score"] is not None else "N/A",
)
b3.metric("Drift Status", week_data["drift_status"])
b4.metric("Valid Scored Features", week_data["similarity_scored_feature_count"])

if week_data["drift_status"] in {"Baseline Building", "Insufficient History"}:
    st.info("Behavioral Similarity Score: N/A. Drift Status: Baseline Building. Four earlier weekly batches and valid feature statistics are required before scoring.")
elif week_data["drift_status"] == "Normal":
    show_severity("Weekly purchase behaviour is consistent with the prior-only historical baseline.", "INFO / NORMAL")
elif week_data["drift_status"] == "Moderate Drift":
    show_severity("Weekly purchase behaviour shows moderate drift from the prior-only historical baseline.", "WARNING")
else:
    show_severity("Weekly purchase behaviour shows major drift from the prior-only historical baseline.", "HIGH")
if week_data["source_window_note"]:
    st.caption(week_data["source_window_note"])
if week_data["excluded_from_similarity_features"]:
    st.caption(week_data["excluded_from_similarity_features"][0]["reason"])

st.subheader("Top Deviating Features")
if week_data["top_deviating_features"]:
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Feature": FEATURE_LABELS[item["feature"]],
                    "Deviation (z-score)": item["z_score"],
                    "Current Batch": item["current"],
                    "Historical Mean": item["baseline_mean"],
                }
                for item in week_data["top_deviating_features"]
            ]
        ),
        use_container_width=True,
        hide_index=True,
        column_config={"Deviation (z-score)": st.column_config.NumberColumn(format="%.2f")},
    )
else:
    st.write("No deviations are ranked while the historical baseline is building.")

st.subheader("Current Batch vs Historical Baseline")
comparison_rows = [
    {
        "Feature": FEATURE_LABELS[key],
        "Current Batch": value["current"],
        "Historical Mean": value["baseline_mean"],
        "Historical Std Dev": value["baseline_stddev"],
        "Deviation (z-score)": value["z_score"],
    }
    for key, value in week_data["feature_comparison"].items()
]
st.dataframe(
    pd.DataFrame(comparison_rows),
    use_container_width=True,
    hide_index=True,
    height=420,
    column_config={
        "Current Batch": st.column_config.NumberColumn(format="%.2f"),
        "Historical Mean": st.column_config.NumberColumn(format="%.2f"),
        "Historical Std Dev": st.column_config.NumberColumn(format="%.2f"),
        "Deviation (z-score)": st.column_config.NumberColumn(format="%.2f"),
    },
)
st.subheader("Historical Behavioral Trend")
trend = pd.DataFrame(
    [{"Week": item["week"], "Behavioral Similarity Score": item["behavioral_similarity_score"]} for item in behavioral]
).set_index("Week")
st.line_chart(trend[["Behavioral Similarity Score"]], height=320)
st.caption("Prototype thresholds: ≥80 Normal, 55–79.99 Moderate Drift, and <55 Major Drift. These are configurable monitoring rules, not universal standards.")

st.divider()
st.header("IMPLEMENTATION 3 — ML Behavioral Anomaly Detection")
st.caption("A Spark ML model learns weekly behavioral groups and flags unusually distant observations.")
if st.button("Run Spark ML Anomaly Model"):
    try:
        with st.spinner("Training KMeans model with Spark ML..."):
            ml_spark = get_spark_session()
            ml_result = run_behavioral_anomaly_model(ml_spark, behavioral)
        st.session_state["ml_anomaly_result"] = ml_result
    except Exception as exc:
        st.error(f"Unable to run the Spark ML model: {type(exc).__name__}: {exc}")

ml_result = st.session_state.get("ml_anomaly_result")
if ml_result:
    x1, x2, x3 = st.columns(3)
    x1.metric("Algorithm", "KMeans k=2")
    x2.metric("Anomalous Weeks", ml_result["anomaly_count"])
    x3.metric("ML Threshold", ml_result["threshold"])
    st.dataframe(pd.DataFrame(ml_result["rows"]), use_container_width=True, hide_index=True)
    st.caption(
        f"Spark ML used {ml_result['feature_count']} standardized behavioral features. "
        "ANOMALOUS means the observation is at or above the prototype 95th-percentile distance threshold."
    )
else:
    st.info("Click 'Run Spark ML Anomaly Model' to generate learned anomaly scores and classifications.")

st.divider()
st.header("IMPLEMENTATION 4 — MongoDB Atlas / NoSQL Report History")
st.caption("Document-based persistence for uploaded-dataset health reports and alert-event history.")
m1, m2, m3 = st.columns(3)
m1.metric("Storage Model", "Document / NoSQL")
m2.metric("Database", "MongoDB Atlas")
m3.metric("Connection", "Configured" if mongodb_settings else "Not Configured")
if mongodb_settings:
    try:
        recent_reports = load_recent_reports(
            mongodb_settings["uri"], mongodb_settings["database"], mongodb_settings["collection"]
        )
        st.success("MongoDB Atlas is configured. Save an uploaded report to add it to this history.")
        if recent_reports:
            st.dataframe(pd.DataFrame(recent_reports), use_container_width=True, hide_index=True)
        else:
            st.info("No uploaded-dataset reports have been saved yet.")
    except Exception:
        st.error("MongoDB is configured but could not be reached. Check the Atlas URI, database user, and network access.")
else:
    st.info("Add MONGODB_URI to Streamlit secrets to enable persistent report history.")
with st.expander("MongoDB document contents"):
    st.write(
        "Each report can store the upload file name, analysis timestamp, schema overview, "
        "data-health score, detected issues, and related alert metadata."
    )

st.divider()
st.header("FINAL MONITORING LAYER — Automated Data Pipeline Monitoring & Alerts")
st.caption("Should the system raise an alert? Batch-triggered / near-real-time architecture; not true streaming.")

# This view follows the selected behavioural week, so all displayed monitoring
# indicators and any applicable alerts refer to the same real batch.
monitoring_week = week_data
monitoring_month = monitoring_week["week"][:7]
applicable = [
    alert
    for alert in alerts
    if alert["batch"] == monitoring_week["week"]
    or (alert["alert_type"] == "Data Health" and alert["batch"] == monitoring_month)
]
highest = max(applicable, key=lambda alert: SEVERITY_RANK[alert["severity"]]) if applicable else None
status = highest["severity"] if highest else "INFO / NORMAL"
a1, a2, a3, a4 = st.columns(4)
a1.metric("Overall Pipeline Status", status)
a2.metric("Current Batch / Week", monitoring_week["week"])
a3.metric(
    "Behavioral Similarity Score",
    f"{monitoring_week['behavioral_similarity_score']}/100"
    if monitoring_week["behavioral_similarity_score"] is not None
    else "N/A",
)
a4.metric("Alert Severity", highest["severity"] if highest else "INFO / NORMAL")

if highest is None:
    show_severity("No applicable warning, high, or critical alert exists for this selected period.", "INFO / NORMAL")
else:
    show_severity(highest["trigger"], highest["severity"])

st.subheader("Alert History")
if alerts:
    alert_rows = [
        {
            "Alert ID": alert["alert_id"],
            "Batch": alert["batch"],
            "Type": alert["alert_type"],
            "Severity": alert["severity"],
            "Similarity": alert["similarity_score"],
            "Top Trigger / Short Reason": short_reason(alert["trigger"]),
            "Generated": alert.get("generated_timestamp"),
        }
        for alert in reversed(alerts)
    ]
    st.dataframe(
        pd.DataFrame(alert_rows),
        use_container_width=True,
        hide_index=True,
        height=300,
        column_config={"Similarity": st.column_config.NumberColumn(format="%.2f")},
    )
    with st.expander("View full alert reasons and details"):
        for alert in reversed(alerts):
            st.markdown(f"**{alert['alert_id']}** — {alert['severity']} — {alert['batch']}")
            st.write(alert["trigger"])
            if alert.get("top_deviating_features"):
                feature_names = [
                    FEATURE_LABELS.get(feature["feature"], feature["feature"])
                    for feature in alert["top_deviating_features"]
                ]
                st.caption("Top deviations: " + ", ".join(feature_names))
else:
    st.success("No alerts have been generated.")

st.divider()
st.subheader("Dataset Source and Transparency")
st.write("Open e-commerce 1.0 — Amazon Purchases public dataset. The active final analysis uses genuine 2022 purchase records only.")
st.write("Survey ResponseID is analysed only as survey-response diversity, not as a verified customer identifier. No cancellation rate is claimed because the source has no confirmed cancellation field.")

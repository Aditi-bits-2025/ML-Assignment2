import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="ML Model Evaluation Dashboard",
    page_icon="📊",
    layout="wide"
)

# --------------------------------------------------
# TYPOGRAPHY & SPACING POLISH
# --------------------------------------------------
st.markdown("""
<style>
.block-container {
    padding: 2rem 2.5rem;
}

h1 {
    font-size: 38px;
    font-weight: 700;
}

h2 {
    font-size: 24px;
}

h3 {
    font-size: 20px;
}

.report-box {
    background:#f8f9fc;
    padding:20px;
    border-radius:12px;
    border:1px solid #e0e4f0;
    font-family: "Courier New", monospace;
    font-size:14px;
    line-height:1.6;
    white-space: pre-wrap;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.title("Machine Learning Model Evaluation Dashboard")
st.caption("Upload dataset, select model, and analyze classification performance")

st.divider()

# --------------------------------------------------
# TWO PANEL LAYOUT
# --------------------------------------------------
left_panel, right_panel = st.columns([1, 3])

# =============================
# LEFT PANEL – CONTROLS
# =============================
with left_panel:
    st.subheader("Configuration")

    uploaded_file = st.file_uploader(
        "Upload Dataset (CSV)",
        type=["csv"]
    )

    selected_model = st.selectbox(
        "Select Model",
        [
            "Logistic Regression",
            "Decision Tree",
            "KNN",
            "Naive Bayes",
            "Random Forest",
            "XGBoost"
        ]
    )

    st.info("Upload dataset and choose model to view evaluation results.")

# =============================
# RIGHT PANEL – RESULTS
# =============================
with right_panel:

    if uploaded_file:

        df = pd.read_csv(uploaded_file)

        if "target" in df.columns:
            X = df.drop(columns=["target"])
            y = df["target"]
        else:
            X = df
            y = None

        model = joblib.load(
            f"model/{selected_model.replace(' ','_')}.pkl"
        )

        predictions = model.predict(X)

        # --------------------------
        # TABS
        # --------------------------
        tab1, tab2, tab3, tab4 = st.tabs(
            ["Dataset Preview", "Performance Metrics", "Confusion Matrix", "Classification Report"]
        )

        # ==========================
        # TAB 1 – PREVIEW
        # ==========================
        with tab1:
            st.subheader("Dataset Preview")
            st.dataframe(df.head(), use_container_width=True)

        # ==========================
        # TAB 2 – METRICS
        # ==========================
        with tab2:
            if y is not None:

                accuracy = accuracy_score(y, predictions)
                precision = precision_score(y, predictions)
                recall = recall_score(y, predictions)
                f1 = f1_score(y, predictions)
                mcc = matthews_corrcoef(y, predictions)

                try:
                    auc = roc_auc_score(y, model.predict_proba(X)[:,1])
                except:
                    auc = 0

                st.subheader("Model Performance Metrics")

                m1, m2, m3 = st.columns(3)
                m4, m5, m6 = st.columns(3)

                m1.metric("Accuracy", f"{accuracy:.4f}")
                m2.metric("AUC Score", f"{auc:.4f}")
                m3.metric("Precision", f"{precision:.4f}")
                m4.metric("Recall", f"{recall:.4f}")
                m5.metric("F1 Score", f"{f1:.4f}")
                m6.metric("MCC Score", f"{mcc:.4f}")

            else:
                st.warning("Target column missing for evaluation.")

        # ==========================
        # TAB 3 – MATRIX
        # ==========================
        with tab3:
            if y is not None:
                st.subheader("Confusion Matrix")

                cm = confusion_matrix(y, predictions)
                cm_percent = cm / cm.sum(axis=1, keepdims=True)

                fig, ax = plt.subplots(figsize=(4.2, 3.2))

                sns.heatmap(
                    cm_percent,
                    annot=cm,
                    fmt="d",
                    cmap="Blues",
                    cbar=False,
                    linewidths=0.5,
                    xticklabels=["Predicted 0", "Predicted 1"],
                    yticklabels=["Actual 0", "Actual 1"],
                    ax=ax
                )

                ax.set_xlabel("Predicted Label")
                ax.set_ylabel("True Label")

                st.pyplot(fig)

            else:
                st.warning("Target column missing.")


        # ==========================
        # TAB 4 – REPORT
        # ==========================
        with tab4:
            if y is not None:
                st.subheader("Classification Report")

                report_dict = classification_report(
                    y,
                    predictions,
                    output_dict=True
                )

                report_df = pd.DataFrame(report_dict).transpose()
                report_df = report_df.round(3)

                # Reorder columns nicely
                report_df = report_df[
                    ["precision", "recall", "f1-score", "support"]
                ]

                st.dataframe(
                    report_df,
                    use_container_width=True,
                    height=360
                )

            else:
                st.warning("Target column missing.")


    else:
        st.subheader("Results Panel")
        st.write("Upload a dataset from the left panel to begin evaluation.")

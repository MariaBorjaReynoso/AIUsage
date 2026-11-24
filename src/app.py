import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Page Configuration & Header
st.set_page_config(
    page_title="AI Usage & Perceived Productivity Analysis",
    page_icon="📊",
    layout="wide",
)

st.title("📊 AI Usage and Perceived Productivity Among Students and Professionals")
st.markdown(
    """
    **Course:** Fall 2026 CSC 405/605/705 Data Science | **Author:** Maria Reynoso  
    **Research Question:** How are daily AI usage hours (`AI_Usage_Hours_Per_Day`), primary task purpose (`AI_Purpose`), 
    and user role (`User_Type`) associated with and predictive of perceived productivity (`Productivity_Score`)?  
    *Note: Because this dataset consists of observational survey responses ($N = 300$), all findings evaluate statistical associations and predictive performance rather than causal effects.*
    """
)

# Stage 2: Data Loading, Cleaning & Preprocessing
DATA_PATH = os.path.join("data", "AI_Usage_and_Impact_on_Students_and_Professionals.csv")


def extract_numeric(val):
    """Strip text units/symbols like '16 yrs', '1.7 hrs', or '$' and convert to float."""
    if pd.isna(val):
        return np.nan
    cleaned = re.sub(r"[^0-9.\-]", "", str(val))
    try:
        return float(cleaned) if cleaned != "" else np.nan
    except ValueError:
        return np.nan


@st.cache_data
def load_and_clean_data(path):
    raw_df = pd.read_csv(path)
    df = raw_df.copy()

    # 1. Remove duplicate User_IDs if any exist
    duplicates_removed = int(df.duplicated(subset=["User_ID"]).sum())
    df = df.drop_duplicates(subset=["User_ID"]).reset_index(drop=True)

    # 2. Clean numeric columns that contain text units or symbols
    numeric_cols = [
        "Age",
        "Monthly_Income",
        "AI_Usage_Hours_Per_Day",
        "Monthly_AI_Cost",
        "Productivity_Score",
        "Accuracy_Rating",
        "Satisfaction_Score",
        "Time_Saved_Hours_Per_Week",
        "Work_or_Study_Hours_Per_Day",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].apply(extract_numeric)

    # 3. Standardize categorical text casing and labels
    if "Gender" in df.columns:
        gender_map = {
            "m": "Male",
            "male": "Male",
            "f": "Female",
            "female": "Female",
        }

        def clean_gender(x):
            if pd.isna(x) or str(x).strip() == "":
                return np.nan
            key = str(x).strip().lower()
            return gender_map.get(key, str(x).strip().title())

        df["Gender"] = df["Gender"].apply(clean_gender)

    for col in ["Country", "User_Type", "Education_Level", "AI_Purpose"]:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: str(x).strip().title() if pd.notna(x) and str(x).strip() != "" else np.nan
            )

    if "AI_Tool" in df.columns:
        tool_map = {
            "chatgpt": "ChatGPT",
            "chat gpt": "ChatGPT",
            "google gemini": "Google Gemini",
            "gemini": "Google Gemini",
            "microsoft copilot": "Microsoft Copilot",
            "copilot": "Microsoft Copilot",
            "claude": "Claude",
            "perplexity ai": "Perplexity AI",
            "perplexity": "Perplexity AI",
            "meta ai": "Meta AI",
            "grammarly": "Grammarly",
        }
        df["AI_Tool"] = df["AI_Tool"].apply(
            lambda x: tool_map.get(str(x).strip().lower(), str(x).strip())
            if pd.notna(x) and str(x).strip() != ""
            else np.nan
        )

    # Record missing values BEFORE imputation
    missing_before = df.isnull().sum()

    # 4. Impute missing values (Median for numeric, Mode for categorical)
    for col in numeric_cols:
        if col in df.columns and df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    cat_cols = [c for c in df.columns if c not in numeric_cols and c != "User_ID"]
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])

    missing_after = df.isnull().sum()

    # 5. Outlier detection summary using 1.5 * IQR rule (retained with justification)
    outlier_rows = []
    for col in ["AI_Usage_Hours_Per_Day", "Productivity_Score", "Time_Saved_Hours_Per_Week", "Monthly_AI_Cost"]:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        count = ((df[col] < lower) | (df[col] > upper)).sum()
        outlier_rows.append(
            {
                "Variable": col,
                "Q1": round(q1, 2),
                "Q3": round(q3, 2),
                "IQR": round(iqr, 2),
                "Lower Fence": round(lower, 2),
                "Upper Fence": round(upper, 2),
                "Outliers Detected": int(count),
            }
        )
    outlier_df = pd.DataFrame(outlier_rows)

    return raw_df, df, duplicates_removed, missing_before, missing_after, outlier_df


raw_df, df, duplicates_removed, missing_before, missing_after, outlier_df = load_and_clean_data(DATA_PATH)

st.header("1. Dataset Overview & Data Preprocessing")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Survey Rows", df.shape[0])
col2.metric("Total Columns", df.shape[1])
col3.metric("Duplicates Removed", duplicates_removed)
col4.metric("Missing Cells Remaining", int(missing_after.sum()))

with st.expander("Preview Raw vs. Cleaned Dataset", expanded=True):
    tab1, tab2 = st.tabs(["Cleaned Data (First 10 Rows)", "Raw Data (First 10 Rows)"])
    with tab1:
        st.dataframe(df.head(10), use_container_width=True)
    with tab2:
        st.dataframe(raw_df.head(10), use_container_width=True)

st.subheader("Missing Value Handling & Outlier Assessment")
c1, c2 = st.columns(2)

with c1:
    missing_summary = pd.DataFrame(
        {"Missing Before": missing_before, "Missing After": missing_after}
    )
    missing_summary = missing_summary[missing_summary["Missing Before"] > 0]
    st.markdown("**Missing Value Imputation Summary**")
    st.dataframe(missing_summary, use_container_width=True)
    st.caption(
        "Justification: Rather than dropping rows on an N = 300 survey dataset (which reduces statistical power), "
        "missing numerical fields were imputed with the median (resistant to skewness) and missing categorical fields with the mode."
    )

with c2:
    st.markdown("**Outlier Detection ($1.5 \\times \\text{IQR}$ Rule)**")
    st.dataframe(outlier_df, use_container_width=True)
    st.caption(
        "Justification: Extreme observations flagged by the 1.5 × IQR fences represent valid heavy-usage or high-spend "
        "respondents rather than data entry errors, so records were retained and evaluated with both parametric and rank-based methods."
    )

# Stage 3: Exploratory Data Analysis (EDA) & Visualizations
os.makedirs("figures", exist_ok=True)
sns.set_theme(style="whitegrid")

st.header("2. Exploratory Data Analysis (EDA)")

st.subheader("Descriptive Statistics & Group-By Aggregations")
tab_desc, tab_role, tab_purpose = st.tabs(
    ["Overall Numeric Summary", "Comparison by User Type", "Comparison by AI Purpose"]
)

with tab_desc:
    st.dataframe(df.describe().T.round(2), use_container_width=True)
    st.caption("Table 1: Descriptive statistics across all numerical survey variables (N = 300).")

with tab_role:
    role_summary = (
        df.groupby("User_Type")[
            ["AI_Usage_Hours_Per_Day", "Productivity_Score", "Time_Saved_Hours_Per_Week", "Satisfaction_Score"]
        ]
        .agg(["count", "mean", "median", "std"])
        .round(2)
    )
    st.dataframe(role_summary, use_container_width=True)
    st.caption("Table 2: Group-by aggregation comparing Students vs. Professionals across key AI metrics.")

with tab_purpose:
    purpose_summary = (
        df.groupby("AI_Purpose")[
            ["AI_Usage_Hours_Per_Day", "Productivity_Score", "Time_Saved_Hours_Per_Week"]
        ]
        .agg(["count", "mean", "median", "std"])
        .round(2)
    )
    st.dataframe(purpose_summary, use_container_width=True)
    st.caption("Table 3: Group-by aggregation of productivity and time saved across primary AI purposes.")

st.subheader("Visualizations & Figure Interpretations")

# Figure 1 Histograms + KDE
fig1, axes1 = plt.subplots(1, 2, figsize=(12, 4.5))
sns.histplot(df["AI_Usage_Hours_Per_Day"], kde=True, bins=15, color="#2b5c8f", ax=axes1[0])
axes1[0].set_title("Distribution of Daily AI Usage Hours")
axes1[0].set_xlabel("AI Usage Hours Per Day")
axes1[0].set_ylabel("Respondent Count")

sns.histplot(df["Productivity_Score"], kde=True, bins=15, color="#2e8b57", ax=axes1[1])
axes1[1].set_title("Distribution of Perceived Productivity Score")
axes1[1].set_xlabel("Productivity Score")
axes1[1].set_ylabel("Respondent Count")
plt.tight_layout()
fig1.savefig("figures/fig1_distributions.png", dpi=300)
st.pyplot(fig1)
st.markdown(
    "**Figure 1 Interpretation:** The histograms and Kernel Density Estimate (KDE) curves illustrate the spread and "
    "shape of daily AI usage and self-reported productivity across all 300 respondents, helping verify whether continuous "
    "variables are approximately symmetric or skewed prior to parametric hypothesis testing."
)

# Figure 2 Box Plots Comparing Students vs. Professionals 
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4.5))
sns.boxplot(data=df, x="User_Type", y="Time_Saved_Hours_Per_Week", hue="User_Type", palette="Set2", legend=False, ax=axes2[0])
axes2[0].set_title("Weekly Time Saved by User Type")
axes2[0].set_xlabel("User Role")
axes2[0].set_ylabel("Time Saved (Hours / Week)")

sns.boxplot(data=df, x="User_Type", y="Productivity_Score", hue="User_Type", palette="Set2", legend=False, ax=axes2[1])
axes2[1].set_title("Perceived Productivity Score by User Type")
axes2[1].set_xlabel("User Role")
axes2[1].set_ylabel("Productivity Score")
plt.tight_layout()
fig2.savefig("figures/fig2_boxplots_user_type.png", dpi=300)
st.pyplot(fig2)
st.markdown(
    "**Figure 2 Interpretation:** Side-by-side box plots compare the median, interquartile range (IQR), and upper/lower "
    "whiskers between Students and Professionals. Overlapping interquartile boxes indicate how similarly or differently "
    "the two cohorts experience weekly time savings and perceived productivity gains."
)

# Figure 3 Scatter Plot & AI Purpose Bar Chart 
fig3, axes3 = plt.subplots(1, 2, figsize=(12, 4.5))
sns.regplot(
    data=df,
    x="AI_Usage_Hours_Per_Day",
    y="Productivity_Score",
    scatter_kws={"alpha": 0.55, "color": "#1f77b4"},
    line_kws={"color": "#d62728"},
    ax=axes3[0],
)
axes3[0].set_title("Daily AI Usage Hours vs. Productivity Score")
axes3[0].set_xlabel("AI Usage Hours Per Day")
axes3[0].set_ylabel("Productivity Score")

purpose_order = df.groupby("AI_Purpose")["Productivity_Score"].mean().sort_values(ascending=False).index
sns.barplot(
    data=df,
    y="AI_Purpose",
    x="Productivity_Score",
    hue="AI_Purpose",
    order=purpose_order,
    palette="viridis",
    errorbar="ci",
    legend=False,
    ax=axes3[1],
)
axes3[1].set_title("Mean Productivity Score by Primary AI Purpose")
axes3[1].set_xlabel("Mean Productivity Score (95% CI)")
axes3[1].set_ylabel("Primary AI Purpose")
plt.tight_layout()
fig3.savefig("figures/fig3_scatter_and_purpose.png", dpi=300)
st.pyplot(fig3)
st.markdown(
    "**Figure 3 Interpretation:** The scatter plot (left) visualizes the linear association between daily AI screen time "
    "and perceived productivity, while the horizontal bar chart (right) ranks average productivity scores across primary "
    "AI task categories (`AI_Purpose`) with 95% confidence intervals."
)

# Figure 4 Correlation Heatmap 
fig4, ax4 = plt.subplots(figsize=(9, 6))
numeric_corr_cols = [
    "Age",
    "Monthly_Income",
    "AI_Usage_Hours_Per_Day",
    "Monthly_AI_Cost",
    "Productivity_Score",
    "Accuracy_Rating",
    "Satisfaction_Score",
    "Time_Saved_Hours_Per_Week",
    "Work_or_Study_Hours_Per_Day",
]
corr_matrix = df[numeric_corr_cols].corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax4)
ax4.set_title("Pearson Correlation Heatmap of Numerical Survey Features")
plt.tight_layout()
fig4.savefig("figures/fig4_correlation_heatmap.png", dpi=300)
st.pyplot(fig4)
st.markdown(
    "**Figure 4 Interpretation:** The correlation heatmap summarizes pairwise linear associations (bounded between -1 and +1) "
    "across all numerical variables, highlighting which usage and rating metrics move together with `Productivity_Score` "
    "and checking for multicollinearity among predictors."
)
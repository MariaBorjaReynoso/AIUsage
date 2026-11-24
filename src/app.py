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

from scipy import stats

# ---------------------------------------------------------
# Stage 4: Statistical Analysis & Hypothesis Testing
# ---------------------------------------------------------
st.header("3. Statistical Analysis & Hypothesis Testing")
st.markdown(
    "All hypothesis tests are evaluated at significance level $\\alpha = 0.05$ on the cleaned dataset ($N = 288$ unique respondents). "
    "Because this is observational survey data, statistically significant results indicate **associations** rather than causal effects."
)

# --- Test 1: Correlation Analysis (Pearson r & Spearman rho) ---
st.subheader("3.1 Correlation Analysis: Daily AI Usage, Productivity & Satisfaction")
st.markdown(
    """
    - **Variables Tested:** `AI_Usage_Hours_Per_Day` vs. `Productivity_Score`, and `AI_Usage_Hours_Per_Day` vs. `Satisfaction_Score`
    - **Null Hypothesis ($H_0$):** There is no linear or monotonic association between daily AI usage hours and productivity or satisfaction ($r = 0, \\rho = 0$).
    - **Alternative Hypothesis ($H_A$):** There is a statistically significant linear or monotonic association ($r \\ne 0, \\rho \\ne 0$).
    - **Test Assumptions & Appropriateness:** Pearson's $r$ evaluates linear relationships on continuous interval variables, while Spearman's rank correlation ($\\rho$) evaluates monotonic relationships without requiring strict normality or sensitivity to outliers.
    """
)

corr_pairs = [
    ("AI_Usage_Hours_Per_Day", "Productivity_Score"),
    ("AI_Usage_Hours_Per_Day", "Satisfaction_Score"),
    ("Productivity_Score", "Satisfaction_Score"),
]
corr_results = []
for x_col, y_col in corr_pairs:
    r_stat, r_p = stats.pearsonr(df[x_col], df[y_col])
    rho_stat, rho_p = stats.spearmanr(df[x_col], df[y_col])
    corr_results.append(
        {
            "Variable Pair": f"{x_col} vs. {y_col}",
            "Pearson r": round(r_stat, 4),
            "Pearson p-value": f"{r_p:.4g}",
            "Variance Explained (r²)": round(r_stat**2, 4),
            "Spearman ρ": round(rho_stat, 4),
            "Spearman p-value": f"{rho_p:.4g}",
            "Decision (α = 0.05)": "Reject H₀" if (r_p < 0.05 or rho_p < 0.05) else "Fail to Reject H₀",
        }
    )

corr_res_df = pd.DataFrame(corr_results)
st.dataframe(corr_res_df, use_container_width=True)

r_up, p_up = stats.pearsonr(df["AI_Usage_Hours_Per_Day"], df["Productivity_Score"])
rho_up, prho_up = stats.spearmanr(df["AI_Usage_Hours_Per_Day"], df["Productivity_Score"])
st.info(
    f"**Contextual Interpretation:** Between `AI_Usage_Hours_Per_Day` and `Productivity_Score`, Pearson $r = {r_up:.3f}$ "
    f"($p = {p_up:.4g}$, effect size $r^2 = {r_up**2:.3f}$) and Spearman $\\rho = {rho_up:.3f}$ ($p = {prho_up:.4g}$). "
    f"{'This indicates a statistically significant association between daily AI usage hours and perceived productivity, though $r^2$ shows that usage hours alone explain only a modest share of variance.' if p_up < 0.05 else 'We fail to reject $H_0$, indicating that simply spending more hours per day using AI is not significantly associated with higher perceived productivity on its own.'}"
)

# --- Test 2: Independent Two-Sample t-Test (Students vs. Professionals) ---
st.subheader("3.2 Independent Two-Sample t-Test: Weekly Time Saved by User Type")

students_ts = df[df["User_Type"] == "Student"]["Time_Saved_Hours_Per_Week"]
profs_ts = df[df["User_Type"] == "Professional"]["Time_Saved_Hours_Per_Week"]

# Assumption checks: Shapiro-Wilk normality & Levene's homogeneity of variance
shapiro_stu_p = stats.shapiro(students_ts)[1]
shapiro_pro_p = stats.shapiro(profs_ts)[1]
levene_stat, levene_p = stats.levene(students_ts, profs_ts)

# Welch's two-sample t-test (robust when variances differ) & Cohen's d effect size
t_stat, t_pval = stats.ttest_ind(students_ts, profs_ts, equal_var=(levene_p >= 0.05))
pooled_sd = np.sqrt(
    ((len(students_ts) - 1) * students_ts.var(ddof=1) + (len(profs_ts) - 1) * profs_ts.var(ddof=1))
    / (len(students_ts) + len(profs_ts) - 2)
)
cohens_d = (students_ts.mean() - profs_ts.mean()) / pooled_sd

st.markdown(
    f"""
    - **Variables Tested:** `User_Type` (Independent variable: Student [$n = {len(students_ts)}$] vs. Professional [$n = {len(profs_ts)}$]) and `Time_Saved_Hours_Per_Week` (Continuous dependent variable).
    - **Null Hypothesis ($H_0$):** Mean weekly time saved is equal for Students and Professionals ($\\mu_{{\\text{{Student}}}} = \\mu_{{\\text{{Professional}}}}$).
    - **Alternative Hypothesis ($H_A$):** Mean weekly time saved differs between Students and Professionals ($\\mu_{{\\text{{Student}}}} \\ne \\mu_{{\\text{{Professional}}}}$).
    - **Test Assumptions & Appropriateness:** A two-sample $t$-test is appropriate for comparing continuous means across two independent groups. Group sample sizes exceed $n = 30$ (satisfying Central Limit Theorem robustness despite Shapiro-Wilk $p = {shapiro_stu_p:.3g}$ and $p = {shapiro_pro_p:.3g}$), and Levene's test ($F = {levene_stat:.3f}, p = {levene_p:.3g}$) informs equal-variance handling.
    """
)

ttest_table = pd.DataFrame(
    [
        {
            "Student Mean ± SD": f"{students_ts.mean():.2f} ± {students_ts.std():.2f}",
            "Professional Mean ± SD": f"{profs_ts.mean():.2f} ± {profs_ts.std():.2f}",
            "t-Statistic": round(t_stat, 4),
            "p-Value": f"{t_pval:.4g}",
            "Effect Size (Cohen's d)": round(cohens_d, 4),
            "Decision (α = 0.05)": "Reject H₀" if t_pval < 0.05 else "Fail to Reject H₀",
        }
    ]
)
st.dataframe(ttest_table, use_container_width=True)

st.info(
    f"**Contextual Interpretation:** The two-sample $t$-test yields $t = {t_stat:.3f}$ ($p = {t_pval:.4g}$) with an effect size "
    f"of Cohen's $d = {cohens_d:.3f}$. {'Because $p < 0.05$, we reject $H_0$: user role is significantly associated with weekly hours saved.' if t_pval < 0.05 else 'Because $p \\ge 0.05$, we fail to reject $H_0$: Students and Professionals report statistically similar weekly time savings from AI tools.'}"
)

# --- Test 3: One-Way ANOVA (Productivity_Score across AI_Purpose) ---
st.subheader("3.3 One-Way ANOVA: Perceived Productivity Across Primary AI Purpose (`AI_Purpose`)")

groups = [group["Productivity_Score"].values for _, group in df.groupby("AI_Purpose")]
purpose_names = [name for name, _ in df.groupby("AI_Purpose")]
levene_anova_stat, levene_anova_p = stats.levene(*groups)
f_stat, anova_pval = stats.f_oneway(*groups)

# Calculate Eta-Squared (eta^2 = SS_between / SS_total) effect size
grand_mean = df["Productivity_Score"].mean()
ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
ss_total = sum((df["Productivity_Score"] - grand_mean) ** 2)
eta_squared = ss_between / ss_total

st.markdown(
    f"""
    - **Variables Tested:** `AI_Purpose` (Categorical factor with {len(purpose_names)} categories: {', '.join(purpose_names)}) and `Productivity_Score` (Continuous dependent variable).
    - **Null Hypothesis ($H_0$):** Mean perceived productivity score is equal across all primary AI task purposes ($\\mu_1 = \\mu_2 = \\dots = \\mu_{{{len(purpose_names)}}}$).
    - **Alternative Hypothesis ($H_A$):** At least one `AI_Purpose` category has a significantly different mean perceived productivity score.
    - **Test Assumptions & Appropriateness:** One-Way ANOVA is appropriate for comparing a continuous mean across three or more independent categorical groups without inflating Type I error. Observations are independent, and Levene's test ($F = {levene_anova_stat:.3f}, p = {levene_anova_p:.3g}$) checks homogeneity of variance across categories.
    """
)

anova_table = pd.DataFrame(
    [
        {
            "Factor Variable": "AI_Purpose",
            "Number of Groups (k)": len(purpose_names),
            "F-Statistic": round(f_stat, 4),
            "p-Value": f"{anova_pval:.4g}",
            "Effect Size (Eta-Squared η²)": round(eta_squared, 4),
            "Decision (α = 0.05)": "Reject H₀" if anova_pval < 0.05 else "Fail to Reject H₀",
        }
    ]
)
st.dataframe(anova_table, use_container_width=True)

st.info(
    f"**Contextual Interpretation:** One-Way ANOVA across `AI_Purpose` produces $F = {f_stat:.3f}$ ($p = {anova_pval:.4g}$) "
    f"with an effect size of $\\eta^2 = {eta_squared:.3f}$ (proportion of variance in productivity associated with task purpose). "
    f"{'Since $p < 0.05$, we reject $H_0$, indicating that perceived productivity differs significantly depending on the primary task for which AI is used.' if anova_pval < 0.05 else 'Since $p \\ge 0.05$, we fail to reject $H_0$, indicating no statistically significant difference in average productivity scores across different primary AI purposes.'}"
)
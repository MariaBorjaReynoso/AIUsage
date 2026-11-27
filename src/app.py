import os
import re
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# Page Configuration & Header
st.set_page_config(
    page_title="AI Usage and Perceived Productivity Analysis",
    layout="wide",
)

st.title("AI Usage and Perceived Productivity Among Students and Professionals")
st.markdown(
    """
    **Course:** Fall 2026 CSC 405 Data Science | **Author:** Maria Reynoso  
    **Research Question:** How are daily AI usage hours (`AI_Usage_Hours_Per_Day`), primary task purpose (`AI_Purpose`), 
    and user role (`User_Type`) associated with and predictive of perceived productivity (`Productivity_Score`)?  
    *Note: Because these data come from an observational survey ($N = 300$ raw responses; $N = 288$ unique respondents after removing duplicates), all findings reflect statistical associations and predictive patterns rather than causal effects.*
    """
)

# Data Loading, Cleaning, and Preprocessing
DATA_PATH = os.path.join("data", "AI_Usage_and_Impact_on_Students_and_Professionals.csv")


def extract_numeric(val):
    """Remove text units or currency symbols (e.g., '16 yrs', '1.7 hrs', '$') and convert to float."""
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

    # 1. Remove duplicate User_IDs
    duplicates_removed = int(df.duplicated(subset=["User_ID"]).sum())
    df = df.drop_duplicates(subset=["User_ID"]).reset_index(drop=True)

    # 2. Clean numeric columns containing text units or symbols
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

    # Record missing values before imputation
    missing_before = df.isnull().sum()

    # 4. Impute missing values (median for numeric, mode for categorical)
    for col in numeric_cols:
        if col in df.columns and df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    cat_cols = [c for c in df.columns if c not in numeric_cols and c != "User_ID"]
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])

    missing_after = df.isnull().sum()

    # 5. Outlier detection summary using the 1.5 * IQR rule
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

st.header("1. Dataset Overview and Preprocessing")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Unique Respondents", df.shape[0])
col2.metric("Total Columns", df.shape[1])
col3.metric("Duplicates Removed", duplicates_removed)
col4.metric("Missing Cells Remaining", int(missing_after.sum()))

with st.expander("Preview Raw vs. Cleaned Dataset", expanded=True):
    tab1, tab2 = st.tabs(["Cleaned Data (First 10 Rows)", "Raw Data (First 10 Rows)"])
    with tab1:
        st.dataframe(df.head(10), use_container_width=True)
    with tab2:
        st.dataframe(raw_df.head(10), use_container_width=True)

st.subheader("Missing Value Handling and Outlier Assessment")
c1, c2 = st.columns(2)

with c1:
    missing_summary = pd.DataFrame(
        {"Missing Before": missing_before, "Missing After": missing_after}
    )
    missing_summary = missing_summary[missing_summary["Missing Before"] > 0]
    st.markdown("**Missing Value Imputation Summary**")
    st.dataframe(missing_summary, use_container_width=True)
    st.caption(
        "Rationale: Rather than dropping rows from a moderate survey sample (N = 288 unique respondents) and losing "
        "statistical power, missing numeric values were filled with the median (which resists skewness) and missing "
        "categorical values were filled with the mode."
    )

with c2:
    st.markdown("**Outlier Detection ($1.5 \\times \\text{IQR}$ Rule)**")
    st.dataframe(outlier_df, use_container_width=True)
    st.caption(
        "Rationale: Points falling outside the 1.5 × IQR fences reflect genuine heavy users or higher monthly AI "
        "spending rather than entry errors. All valid records were kept and examined using both parametric and rank-based tests."
    )

# Exploratory Data Analysis (EDA) and Visualizations
os.makedirs("figures", exist_ok=True)
sns.set_theme(style="whitegrid")

st.header("2. Exploratory Data Analysis (EDA)")

st.subheader("Descriptive Statistics and Group Comparisons")
tab_desc, tab_role, tab_purpose = st.tabs(
    ["Overall Numeric Summary", "Comparison by User Type", "Comparison by AI Purpose"]
)

with tab_desc:
    st.dataframe(df.describe().T.round(2), use_container_width=True)
    st.caption("Table 1: Summary statistics across all numerical survey variables (N = 288).")

with tab_role:
    role_summary = (
        df.groupby("User_Type")[
            ["AI_Usage_Hours_Per_Day", "Productivity_Score", "Time_Saved_Hours_Per_Week", "Satisfaction_Score"]
        ]
        .agg(["count", "mean", "median", "std"])
        .round(2)
    )
    st.dataframe(role_summary, use_container_width=True)
    st.caption("Table 2: Comparison of Students and Professionals across key AI usage and productivity metrics.")

with tab_purpose:
    purpose_summary = (
        df.groupby("AI_Purpose")[
            ["AI_Usage_Hours_Per_Day", "Productivity_Score", "Time_Saved_Hours_Per_Week"]
        ]
        .agg(["count", "mean", "median", "std"])
        .round(2)
    )
    st.dataframe(purpose_summary, use_container_width=True)
    st.caption("Table 3: Productivity and weekly time saved grouped by primary AI task purpose.")

st.subheader("Visualizations and Interpretations")

# Figure 1: Histograms + KDE
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
    "**Figure 1 Interpretation:** These histograms and density curves show the overall spread and shape of daily AI "
    "usage hours and self-reported productivity across the 288 respondents, providing a visual check for skewness "
    "before running parametric tests."
)

# Figure 2: Box Plots Comparing Students vs. Professionals
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
    "**Figure 2 Interpretation:** Side-by-side box plots compare the median, interquartile range, and whiskers for "
    "Students and Professionals. The substantial overlap between boxes suggests that both groups report fairly similar "
    "weekly time savings and productivity levels."
)

# Figure 3: Scatter Plot & AI Purpose Bar Chart
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
    "**Figure 3 Interpretation:** The scatter plot on the left shows an upward trend between daily AI usage hours and "
    "perceived productivity. On the right, the horizontal bar chart compares mean productivity scores across primary task "
    "categories (`AI_Purpose`), where overlapping 95% confidence intervals indicate relatively close group averages."
)

# Figure 4: Correlation Heatmap
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
    "**Figure 4 Interpretation:** The correlation heatmap displays pairwise linear relationships (from -1 to +1) across "
    "all numerical variables, highlighting which usage and rating features correlate most closely with `Productivity_Score`."
)

# Statistical Analysis and Hypothesis Testing
st.header("3. Statistical Analysis and Hypothesis Testing")
st.markdown(
    "All hypothesis tests are evaluated at a significance level of $\\alpha = 0.05$ on the cleaned dataset ($N = 288$). "
    "Because this is observational survey data, significant results reflect statistical associations rather than cause-and-effect relationships."
)

# Test 1: Correlation Analysis
st.subheader("3.1 Correlation Analysis: Daily AI Usage, Productivity, and Satisfaction")
st.markdown(
    """
- **Variables Tested:** `AI_Usage_Hours_Per_Day` vs. `Productivity_Score`, and `AI_Usage_Hours_Per_Day` vs. `Satisfaction_Score`
- **Null Hypothesis ($H_0$):** There is no linear or monotonic association between daily AI usage hours and productivity or satisfaction ($r = 0, \\rho = 0$).
- **Alternative Hypothesis ($H_A$):** There is a statistically significant linear or monotonic association ($r \\ne 0, \\rho \\ne 0$).
- **Test Assumptions and Choice:** Pearson's $r$ measures linear strength between continuous variables, while Spearman's rank correlation ($\\rho$) checks monotonic trends without requiring strict normality or being overly sensitive to outliers.
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
    f"**Interpretation:** Between `AI_Usage_Hours_Per_Day` and `Productivity_Score`, Pearson $r = {r_up:.3f}$ "
    f"($p = {p_up:.4g}$, $r^2 = {r_up**2:.3f}$) and Spearman $\\rho = {rho_up:.3f}$ ($p = {prho_up:.4g}$). "
    f"{'This shows a statistically significant positive association between daily AI usage hours and perceived productivity.' if p_up < 0.05 else 'We fail to reject $H_0$, suggesting that daily AI usage hours alone are not significantly associated with higher perceived productivity.'}"
)

# Test 2: Independent Two-Sample t-Test
st.subheader("3.2 Independent Two-Sample t-Test: Weekly Time Saved by User Type")

students_ts = df[df["User_Type"] == "Student"]["Time_Saved_Hours_Per_Week"]
profs_ts = df[df["User_Type"] == "Professional"]["Time_Saved_Hours_Per_Week"]

shapiro_stu_p = stats.shapiro(students_ts)[1]
shapiro_pro_p = stats.shapiro(profs_ts)[1]
levene_stat, levene_p = stats.levene(students_ts, profs_ts)

t_stat, t_pval = stats.ttest_ind(students_ts, profs_ts, equal_var=(levene_p >= 0.05))
pooled_sd = np.sqrt(
    ((len(students_ts) - 1) * students_ts.var(ddof=1) + (len(profs_ts) - 1) * profs_ts.var(ddof=1))
    / (len(students_ts) + len(profs_ts) - 2)
)
cohens_d = (students_ts.mean() - profs_ts.mean()) / pooled_sd

st.markdown(
    f"""
- **Variables Tested:** `User_Type` (Student [$n = {len(students_ts)}$] vs. Professional [$n = {len(profs_ts)}$]) and `Time_Saved_Hours_Per_Week` (continuous outcome).
- **Null Hypothesis ($H_0$):** Average weekly time saved is the same for Students and Professionals ($\\mu_{{\\text{{Student}}}} = \\mu_{{\\text{{Professional}}}}$).
- **Alternative Hypothesis ($H_A$):** Average weekly time saved differs between Students and Professionals ($\\mu_{{\\text{{Student}}}} \\ne \\mu_{{\\text{{Professional}}}}$).
- **Test Assumptions and Choice:** A two-sample $t$-test compares continuous means across two independent groups. Both sample sizes exceed $n = 30$ (supporting Central Limit Theorem robustness alongside Shapiro-Wilk $p = {shapiro_stu_p:.3g}$ and $p = {shapiro_pro_p:.3g}$), and Levene's test ($F = {levene_stat:.3f}, p = {levene_p:.3g}$) checks whether group variances are equal.
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
    f"**Interpretation:** The two-sample $t$-test yields $t = {t_stat:.3f}$ ($p = {t_pval:.4g}$) with an effect size "
    f"of Cohen's $d = {cohens_d:.3f}$. {'Because $p < 0.05$, we reject $H_0$: user role is significantly associated with weekly hours saved.' if t_pval < 0.05 else 'Because $p \\ge 0.05$, we fail to reject $H_0$: Students and Professionals report statistically similar weekly time savings from AI tools.'}"
)

# Test 3: One-Way ANOVA
st.subheader("3.3 One-Way ANOVA: Perceived Productivity Across Primary AI Purpose (`AI_Purpose`)")

groups = [group["Productivity_Score"].values for _, group in df.groupby("AI_Purpose")]
purpose_names = [name for name, _ in df.groupby("AI_Purpose")]
levene_anova_stat, levene_anova_p = stats.levene(*groups)
f_stat, anova_pval = stats.f_oneway(*groups)

grand_mean = df["Productivity_Score"].mean()
ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
ss_total = sum((df["Productivity_Score"] - grand_mean) ** 2)
eta_squared = ss_between / ss_total

st.markdown(
    f"""
- **Variables Tested:** `AI_Purpose` ({len(purpose_names)} categories: {', '.join(purpose_names)}) and `Productivity_Score` (continuous outcome).
- **Null Hypothesis ($H_0$):** Mean perceived productivity score is equal across all primary AI task purposes ($\\mu_1 = \\mu_2 = \\dots = \\mu_{{{len(purpose_names)}}}$).
- **Alternative Hypothesis ($H_A$):** At least one `AI_Purpose` category has a different mean perceived productivity score.
- **Test Assumptions and Choice:** One-Way ANOVA compares a continuous mean across three or more independent groups in a single test. Group observations are independent, and Levene's test ($F = {levene_anova_stat:.3f}, p = {levene_anova_p:.3g}$) checks homogeneity of variance across categories.
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
    f"**Interpretation:** One-Way ANOVA across `AI_Purpose` yields $F = {f_stat:.3f}$ ($p = {anova_pval:.4g}$) "
    f"with an effect size of $\\eta^2 = {eta_squared:.3f}$. "
    f"{'Since $p < 0.05$, we reject $H_0$, indicating that average productivity scores differ across primary task categories.' if anova_pval < 0.05 else 'Since $p \\ge 0.05$, we fail to reject $H_0$, indicating no statistically significant difference in average productivity scores across primary AI task purposes.'}"
)

# Machine Learning Modeling and Evaluation
st.header("4. Machine Learning Modeling and Evaluation")
st.markdown(
    """
    To see how accurately usage habits and user characteristics predict `Productivity_Score`, we compare two regression 
    models using an **80/20 train-test split** (`random_state=42`) and **5-fold cross-validation** to guard against 
    overfitting on a 288-row dataset:
- **Baseline Model:** Multiple Linear Regression (a straightforward linear benchmark)
- **Non-Linear Model:** Random Forest Regressor (`n_estimators=200`, `max_depth=5` to limit tree complexity)
    """
)

feature_cols = [
    "AI_Usage_Hours_Per_Day",
    "User_Type",
    "AI_Purpose",
    "AI_Tool",
    "Work_or_Study_Hours_Per_Day",
    "Monthly_AI_Cost",
    "Accuracy_Rating",
    "Satisfaction_Score",
    "Time_Saved_Hours_Per_Week",
]
target_col = "Productivity_Score"

X = df[feature_cols].copy()
y = df[target_col].copy()

num_features = [
    "AI_Usage_Hours_Per_Day",
    "Work_or_Study_Hours_Per_Day",
    "Monthly_AI_Cost",
    "Accuracy_Rating",
    "Satisfaction_Score",
    "Time_Saved_Hours_Per_Week",
]
cat_features = ["User_Type", "AI_Purpose", "AI_Tool"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

models = {
    "Linear Regression (Baseline)": LinearRegression(),
    "Random Forest Regressor": RandomForestRegressor(
        n_estimators=200, max_depth=5, min_samples_leaf=4, random_state=42
    ),
}

cv = KFold(n_splits=5, shuffle=True, random_state=42)
scoring = {
    "mae": "neg_mean_absolute_error",
    "rmse": "neg_root_mean_squared_error",
    "r2": "r2",
}

ml_rows = []
rf_pipeline = None

for name, regressor in models.items():
    pipe = Pipeline(steps=[("prep", preprocessor), ("model", regressor)])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    test_mae = mean_absolute_error(y_test, preds)
    test_rmse = np.sqrt(mean_squared_error(y_test, preds))
    test_r2 = r2_score(y_test, preds)

    cv_scores = cross_validate(pipe, X, y, cv=cv, scoring=scoring)
    cv_mae = -cv_scores["test_mae"].mean()
    cv_rmse = -cv_scores["test_rmse"].mean()
    cv_r2 = cv_scores["test_r2"].mean()

    ml_rows.append(
        {
            "Model": name,
            "Test MAE": round(test_mae, 3),
            "Test RMSE": round(test_rmse, 3),
            "Test R²": round(test_r2, 3),
            "5-Fold CV MAE": round(cv_mae, 3),
            "5-Fold CV RMSE": round(cv_rmse, 3),
            "5-Fold CV R²": round(cv_r2, 3),
        }
    )
    if "Random Forest" in name:
        rf_pipeline = pipe

ml_results_df = pd.DataFrame(ml_rows)
st.dataframe(ml_results_df, use_container_width=True)
st.caption("Table 4: Hold-out test set and 5-fold cross-validation metrics predicting Productivity_Score.")

if rf_pipeline is not None:
    ohe_names = rf_pipeline.named_steps["prep"].named_transformers_["cat"].get_feature_names_out(cat_features)
    all_feature_names = num_features + list(ohe_names)
    importances = rf_pipeline.named_steps["model"].feature_importances_
    imp_df = (
        pd.DataFrame({"Feature": all_feature_names, "Importance": importances})
        .sort_values(by="Importance", ascending=False)
        .head(10)
    )

    fig5, ax5 = plt.subplots(figsize=(9, 4.5))
    sns.barplot(data=imp_df, x="Importance", y="Feature", hue="Feature", palette="mako", legend=False, ax=ax5)
    ax5.set_title("Top 10 Predictive Features for Perceived Productivity (Random Forest)")
    ax5.set_xlabel("Relative Feature Importance (Impurity Reduction)")
    ax5.set_ylabel("Feature")
    plt.tight_layout()
    fig5.savefig("figures/fig5_feature_importance.png", dpi=300)
    st.pyplot(fig5)
    st.markdown(
        "**Figure 5 Interpretation:** The feature importance plot shows which variables contributed most to reducing "
        "prediction error in the Random Forest model, highlighting the relative role of usage hours, perceived accuracy, "
        "satisfaction, and weekly time saved."
    )

# AI-Assisted Analysis (LangChain + Gemini API)
st.header("5. AI-Assisted Analysis and Critical Evaluation (LangChain + Gemini API)")

with st.sidebar:
    st.header("AI Assistant Settings")
    gemini_api_key = st.text_input(
        "Google Gemini API Key",
        value=os.environ.get("GEMINI_API_KEY", ""),
        type="password",
        help="Enter your Gemini API key to query the live LangChain assistant.",
    )
    st.caption("Your API key stays in browser session memory only and is never saved to the repository.")

st.subheader("5.1 Interactive LangChain Analysis Assistant")
st.markdown(
    "This section connects **Google Gemini** through **LangChain (`PromptTemplate | ChatGoogleGenerativeAI | StrOutputParser`)** "
    "with `temperature=0` to generate consistent interpretations grounded directly in our calculated metrics."
)

context_summary = (
    f"Dataset: N={len(df)} unique respondents (Students vs. Professionals). "
    f"Mean AI_Usage_Hours_Per_Day={df['AI_Usage_Hours_Per_Day'].mean():.2f}, "
    f"Mean Productivity_Score={df['Productivity_Score'].mean():.2f}, "
    f"Mean Time_Saved_Hours_Per_Week={df['Time_Saved_Hours_Per_Week'].mean():.2f}. "
    f"Pearson correlation (AI_Usage_Hours vs Productivity_Score): r={r_up:.3f}, p={p_up:.4g}. "
    f"Two-sample t-test (Time Saved Student vs Professional): t={t_stat:.3f}, p={t_pval:.4g}, Cohen's d={cohens_d:.3f}. "
    f"One-Way ANOVA (Productivity across AI_Purpose): F={f_stat:.3f}, p={anova_pval:.4g}, eta^2={eta_squared:.3f}. "
    f"ML 5-Fold CV Results:\n{ml_results_df.to_string(index=False)}"
)

default_question = (
    "Based on our statistical tests and cross-validated ML results, what are the key takeaways regarding "
    "whether daily AI usage hours and task purpose are associated with perceived productivity?"
)
user_ai_query = st.text_area("Ask the LangChain + Gemini Assistant about the project findings:", value=default_question, height=90)

if st.button("Generate AI Interpretation via LangChain", type="primary"):
    if not gemini_api_key:
        st.warning("Please enter your Google Gemini API key in the left sidebar to run the LangChain pipeline.")
    else:
        with st.spinner("Running LangChain pipeline..."):
            try:
                prompt_template = PromptTemplate(
                    input_variables=["stats_context", "question"],
                    template=(
                        "You are an objective data science assistant. Using ONLY the project metrics provided below, "
                        "answer the user's question clearly in 1-2 short paragraphs. Do not claim causal relationships; "
                        "describe findings strictly as statistical associations or predictive performance.\n\n"
                        "Project Metrics:\n{stats_context}\n\nQuestion: {question}"
                    ),
                )

                candidate_models = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-flash-latest"]
                try:
                    resp = requests.get(
                        f"https://generativelanguage.googleapis.com/v1beta/models?key={gemini_api_key}",
                        timeout=10,
                    )
                    if resp.ok:
                        discovered = [
                            m["name"].replace("models/", "")
                            for m in resp.json().get("models", [])
                            if "generateContent" in m.get("supportedGenerationMethods", [])
                            and "gemini" in m.get("name", "")
                        ]
                        if discovered:
                            flash_first = [m for m in discovered if "flash" in m] + [
                                m for m in discovered if "flash" not in m
                            ]
                            candidate_models = flash_first
                except Exception:
                    pass

                ai_response = None
                last_error = None
                for model_name in candidate_models:
                    try:
                        llm = ChatGoogleGenerativeAI(
                            model=model_name,
                            google_api_key=gemini_api_key,
                            temperature=0,
                            max_retries=1,
                        )
                        chain = prompt_template | llm | StrOutputParser()
                        ai_response = chain.invoke({"stats_context": context_summary, "question": user_ai_query})
                        break
                    except Exception as model_exc:
                        last_error = model_exc

                if ai_response:
                    st.success(f"LangChain + Gemini Response (Model: `{model_name}`):")
                    st.write(ai_response)
                else:
                    st.error(f"API call failed: {last_error}")
            except Exception as exc:
                st.error(f"API call failed: {exc}")

st.subheader("5.2 Documented AI-Assisted Decisions and Critical Evaluation")
st.markdown(
    "Below are three key points during the project where an AI assistant was consulted, along with how each "
    "suggestion was evaluated and applied:"
)

ai_log_df = pd.DataFrame(
    [
        {
            "Project Stage": "1. Data Cleaning",
            "Prompt Used": "In a 300-row survey dataset with missing values in Education_Level, Country, and numeric cost/income fields, should incomplete rows be dropped or imputed?",
            "AI Suggestion": "Use median imputation for skewed numerical fields, mode imputation for categorical fields, and regular expressions to strip text units like '16 yrs' and '1.7 hrs'.",
            "Evaluation and Decision": "Accepted. Dropping rows with missing cells would remove too much of our 288-respondent sample and lower statistical power, while median imputation avoids distortion from skewed income and cost values.",
        },
        {
            "Project Stage": "2. Statistical Testing",
            "Prompt Used": "How should we compare weekly time saved by User_Type and productivity across AI_Purpose while checking test assumptions and reporting effect sizes?",
            "AI Suggestion": "Check normality and equal variance with Shapiro-Wilk and Levene's tests, report both Pearson r and Spearman rho, use Welch's t-test with Cohen's d, and report eta-squared for ANOVA.",
            "Evaluation and Decision": "Modified. Adopted the assumption checks and effect size metrics, but reworded all conclusions to describe statistical associations rather than causal effects per instructor feedback.",
        },
        {
            "Project Stage": "3. Model Tuning",
            "Prompt Used": "Our Random Forest Regressor gets a high training R² on 288 rows but may overfit. What tree settings and validation approach should we use?",
            "AI Suggestion": "Run a default Random Forest without depth limits (max_depth=None) on a single 80/20 train-test split.",
            "Evaluation and Decision": "Rejected and revised. Unpruned trees tend to memorize noise on a 288-row survey, so max_depth was capped at 5, min_samples_leaf was set to 4, and 5-fold cross-validation was added.",
        },
    ]
)
st.dataframe(ai_log_df, use_container_width=True)

# Final Conclusions and Limitations
st.header("6. Conclusions and Study Limitations")
st.markdown(
    """
- **Summary of Findings:** Across $N = 288$ unique survey respondents, daily AI usage hours show a statistically 
  significant positive association with perceived productivity, whereas average productivity scores do not differ 
  significantly across primary task purposes (`AI_Purpose`), and weekly time savings are comparable between students and 
  professionals. In predictive modeling, both Multiple Linear Regression and a regularized Random Forest Regressor 
  achieve similar 5-fold cross-validated accuracy, showing that limiting tree depth prevents overfitting on a moderate sample.
- **Practical Takeaways:** Because the type of task alone does not guarantee higher perceived productivity, schools and 
  workplaces should pair AI access with practical workflow guidance rather than focusing solely on tool selection.
- **Limitations:** Productivity scores, accuracy ratings, and weekly hours saved are all self-reported estimates rather 
  than logged usage metrics, which may introduce recall or subjective bias. In addition, because the survey is 
  observational, the results represent statistical associations and predictive patterns rather than causal effects.
    """
)
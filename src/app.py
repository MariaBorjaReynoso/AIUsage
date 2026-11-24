import matplotlib.pyplot as plt
import seaborn as sns

# Figures directory exists for reproducible export
os.makedirs("figures", exist_ok=True)
sns.set_theme(style="whitegrid")

# Exploratory Data Analysis (EDA) & Visualizations 
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

# Figure 1: Histograms + KDE of Daily AI Usage & Productivity Score
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

# Box Plots Comparing Students vs. Professionals 
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4.5))
sns.boxplot(data=df, x="User_Type", y="Time_Saved_Hours_Per_Week", palette="Set2", ax=axes2[0])
axes2[0].set_title("Weekly Time Saved by User Type")
axes2[0].set_xlabel("User Role")
axes2[0].set_ylabel("Time Saved (Hours / Week)")

sns.boxplot(data=df, x="User_Type", y="Productivity_Score", palette="Set2", ax=axes2[1])
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

# Scatter Plot with Regression Trendline & AI Purpose Bar Chart 
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
    order=purpose_order,
    palette="viridis",
    errorbar="ci",
    ax=axes3[1],
)
axes3[1].set_title("Mean Productivity Score by Primary AI Purpose")
axes3[1].set_xlabel("Mean Productivity Score (95% CI)")
axes3[1].set_ylabel("Primary AI Purpose")
plt.tight_layout()
fig3.savefig("figures/fig3_scatter_and_purpose.png", dpi=300)
st.pyplot(fig3)
st.markdown(
    "**The scatter plot" "Interpretation:** The scatter plot (left) visualizes the linear association between daily AI screen time "
    "and perceived productivity, while the horizontal bar chart (right) ranks average productivity scores across primary "
    "AI task categories (`AI_Purpose`) with 95% confidence intervals."
)

#  Correlation Heatmap 
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
    "**HeatmapInterpretation:** The correlation heatmap summarizes pairwise linear associations (bounded between -1 and +1) "
    "across all numerical variables, highlighting which usage and rating metrics move together with `Productivity_Score` "
    "and checking for multicollinearity among predictors."
)
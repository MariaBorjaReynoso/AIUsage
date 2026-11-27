import os
import shutil
from fpdf import FPDF

class ReportPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(100, 100, 100)
        self.cell(
            0, 6,
            "AI Usage and Perceived Productivity Among Students and Professionals | Fall 2026 CSC 405",
            align="R", new_x="LMARGIN", new_y="NEXT"
        )
        self.ln(2)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, f"Page {self.page_no()}", align="C")

    def section_title(self, title):
        self.ln(4)
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(15, 32, 67)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(200, 210, 225)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(3)

    def body_text(self, text, bold=False):
        self.set_font("Helvetica", "B" if bold else "", 10)
        self.set_text_color(35, 35, 35)
        self.multi_cell(0, 5.5, text)
        self.ln(2)

pdf = ReportPDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()

# Title Block
pdf.set_font("Helvetica", "B", 17)
pdf.set_text_color(15, 32, 67)
pdf.multi_cell(0, 8, "AI Usage and Perceived Productivity Among Students and Professionals")
pdf.ln(2)
pdf.body_text("Course: Fall 2026 CSC 405 Data Science | Author: Maria Reynoso", bold=True)
pdf.body_text(
    "Research Question: How are daily AI usage hours (AI_Usage_Hours_Per_Day), primary task purpose "
    "(AI_Purpose), and user role (User_Type) associated with and predictive of perceived productivity (Productivity_Score)?\n"
    "Note: Because these data come from an observational survey (N = 300 raw responses; N = 288 unique respondents "
    "after removing duplicates), all findings reflect statistical associations and predictive patterns rather than causal effects."
)

# Section 1
pdf.section_title("1. Dataset Overview and Preprocessing")
pdf.body_text(
    "- Unique Respondents: 288 | Total Columns: 19 | Duplicates Removed: 12 | Missing Cells Remaining: 0\n"
    "- Missing Value Handling: Missing numeric values were imputed with the median (resistant to skewness), and missing categorical values were imputed with the mode.\n"
    "- Outlier Assessment (1.5 x IQR Rule): Points outside the 1.5 x IQR fences represent genuine heavy AI users or higher monthly AI spend rather than data entry errors; all 288 valid records were retained."
)

# Section 2 + Figures
pdf.section_title("2. Exploratory Data Analysis (EDA)")
pdf.body_text(
    "Descriptive statistics and visualizations examine the distribution of daily AI usage hours, productivity scores, "
    "group comparisons between Students (n = 151) and Professionals (n = 137), and pairwise linear correlations."
)

figure_files = [
    ("figures/fig1_distributions.png", "Figure 1: Distributions of daily AI usage hours and perceived productivity scores (N = 288)."),
    ("figures/fig2_boxplots_user_type.png", "Figure 2: Comparison of weekly time saved and productivity scores by user role (Student vs. Professional)."),
    ("figures/fig3_scatter_and_purpose.png", "Figure 3: Daily AI usage vs. productivity trend and mean productivity across primary task categories."),
    ("figures/fig4_correlation_heatmap.png", "Figure 4: Pairwise Pearson correlation heatmap across numerical survey variables.")
]

for path, caption in figure_files:
    if os.path.exists(path):
        if pdf.get_y() > 195:
            pdf.add_page()
        pdf.image(path, w=175)
        pdf.ln(1)
        pdf.set_font("Helvetica", "I", 9)
        pdf.cell(0, 5, caption, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(3)

# Section 3
pdf.section_title("3. Statistical Analysis and Hypothesis Testing (alpha = 0.05)")
pdf.body_text(
    "3.1 Correlation Analysis (AI_Usage_Hours_Per_Day vs. Productivity_Score):\n"
    "- Pearson r = 0.636 (p = 4.305e-34, r^2 = 0.405) and Spearman rho = 0.772 (p = 2.94e-58). Reject H0: statistically significant positive association between daily AI usage hours and perceived productivity.\n\n"
    "3.2 Independent Two-Sample t-Test (Weekly Time Saved: Students [n=151] vs. Professionals [n=137]):\n"
    "- Welch's t = 1.033 (p = 0.3025, Cohen's d = 0.121). Fail to reject H0: both groups report statistically similar weekly time savings.\n\n"
    "3.3 One-Way ANOVA (Productivity_Score Across 8 AI_Purpose Categories):\n"
    "- F = 1.047 (p = 0.3988, eta^2 = 0.025). Fail to reject H0: mean perceived productivity does not differ significantly across task purposes."
)

# Section 4 + Figure 5
pdf.section_title("4. Machine Learning Modeling and Evaluation")
pdf.body_text(
    "Using an 80/20 train-test split (random_state=42) and 5-fold cross-validation predicting Productivity_Score:\n"
    "- Baseline Multiple Linear Regression: Test R^2 = 0.651, Test MAE = 0.830\n"
    "- Regularized Random Forest Regressor (n_estimators=200, max_depth=5): Test R^2 = 0.704, Test MAE = 0.764, 5-Fold CV R^2 = 0.618"
)
if os.path.exists("figures/fig5_feature_importance.png"):
    if pdf.get_y() > 195:
        pdf.add_page()
    pdf.image("figures/fig5_feature_importance.png", w=175)
    pdf.ln(1)
    pdf.set_font("Helvetica", "I", 9)
    pdf.cell(0, 5, "Figure 5: Random Forest feature importance ranking for predicting Productivity_Score.", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

# Section 5
pdf.section_title("5. AI-Assisted Analysis and Critical Evaluation (LangChain + Gemini API)")
pdf.body_text(
    "5.1 Live LangChain + Gemini Interpretation (Model: gemini-flash-lite-latest, temperature=0):\n"
    "\"Based on the provided metrics, daily AI usage hours show a strong positive statistical association with perceived productivity, "
    "supported by a significant Pearson correlation of r = 0.636 (p = 4.305e-34). In contrast, task purpose is not significantly associated "
    "with productivity, as indicated by a One-Way ANOVA yield of F = 1.047 (p = 0.3988) and a small effect size (eta^2 = 0.025). "
    "Regarding predictive performance, machine learning models utilizing these features demonstrate moderate to good capability in predicting "
    "productivity scores. The baseline Linear Regression achieved a Test R^2 of 0.651 (Test MAE 0.830), while the Random Forest Regressor "
    "showed a slight improvement in test performance with an R^2 of 0.704 (Test MAE 0.764) and a 5-Fold CV R^2 of 0.618.\""
)
pdf.body_text(
    "5.2 Documented AI-Assisted Decisions & Critical Evaluation:\n"
    "1. Data Preprocessing: Verified AI suggestion for median/mode imputation and 1.5 x IQR outlier retention against course rules so N = 288 valid records were preserved.\n"
    "2. Hypothesis Testing: Checked normality (Shapiro-Wilk) and variance equality (Levene's test) before reporting Welch's t-test and non-parametric Spearman rank correlation.\n"
    "3. Model Regularization: Constrained Random Forest depth (max_depth=5) and validated with 5-fold CV to prevent overfitting on a 288-row dataset."
)

# Section 6
pdf.section_title("6. Conclusions and Study Limitations")
pdf.body_text(
    "- Summary of Findings: Daily AI usage hours correlate positively with perceived productivity (r = 0.636), whereas productivity scores do not differ significantly across primary task purposes, and weekly time savings are comparable between students and professionals.\n"
    "- Practical Takeaways: Schools and workplaces should pair AI tool access with structured workflow guidance rather than focusing solely on tool or task selection.\n"
    "- Limitations: Productivity scores, accuracy ratings, and weekly hours saved are self-reported survey estimates rather than logged metrics, and observational data reflect statistical associations rather than causal effects.\n"
    "- GitHub Code Repository: https://github.com/MariaBorjaReynoso/AIUsage"
)

desktop_path = r"C:\Users\Maria Reynoso\Desktop\Final_Project_Report.pdf"
os.makedirs("report", exist_ok=True)
repo_report_path = "report/Final_Project_Report.pdf"

pdf.output(desktop_path)
shutil.copy(desktop_path, repo_report_path)
print(f"Successfully generated: {desktop_path} and {repo_report_path}")
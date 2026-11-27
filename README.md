# AIUsage

Data science analysis and interactive Streamlit dashboard exploring how daily AI usage hours, task purpose, and user role relate to perceived productivity.

## Summary of Findings
* People who use AI for more hours each day tend to report higher productivity ($r = 0.636, p < 0.001$).
* Average productivity scores are pretty consistent across different task types, and students and professionals report similar weekly time savings.
* A Random Forest model ($R^2 = 0.704$) predicted productivity scores better than a basic Linear Regression model ($R^2 = 0.651$).

## Project Files
* `data/` — Cleaned survey dataset (288 responses, 19 columns)
* `src/app.py` — Streamlit dashboard with LangChain + Gemini integration
* `figures/` — Charts and feature importance plots
* `report/Final_Project_Report.pdf` — Final project report
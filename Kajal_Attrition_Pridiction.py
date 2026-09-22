"""
Employee Attrition Prediction
Single-file app: backend (scikit-learn) + frontend (Gradio)
Dataset: Attrition.csv.csv
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io, base64, os


from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, accuracy_score, roc_curve
)
from sklearn.pipeline import Pipeline
import gradio as gr
from gradio import themes as gr_themes

# ─────────────────────────────────────────────
# 1.  DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────

CSV_PATH = os.path.join(os.path.dirname(__file__), "Attrition.csv.csv")

def load_and_prepare():
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")

    # Drop columns with no variance / not useful
    df.drop(columns=["EmployeeCount", "Over18", "StandardHours"], inplace=True, errors="ignore")

    # Encode target
    df["Attrition_Binary"] = (df["Attrition"] == "Yes").astype(int)

    # Identify column types
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    cat_cols = [c for c in cat_cols if c != "Attrition"]

    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    feature_cols = (
        [c for c in df.columns if c.endswith("_enc")]
        + df.select_dtypes(include=[np.number]).columns.tolist()
    )
    feature_cols = [c for c in feature_cols if c not in ("Attrition_Binary",)]
    feature_cols = list(dict.fromkeys(feature_cols))  # deduplicate preserving order

    X = df[feature_cols]
    y = df["Attrition_Binary"]

    return df, X, y, feature_cols, cat_cols, encoders


df_raw, X, y, FEATURE_COLS, CAT_COLS, ENCODERS = load_and_prepare()

# ─────────────────────────────────────────────
# 2.  MODEL TRAINING
# ─────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = np.array(model.predict_proba(X_test))[:, 1]

ACCURACY = accuracy_score(y_test, y_pred)
ROC_AUC = roc_auc_score(y_test, y_proba)
CV_SCORES = cross_val_score(model, X, y, cv=5, scoring="roc_auc")
CLF_REPORT = classification_report(
    y_test,
    y_pred,
    target_names=["Stayed", "Left"]
)

# Feature importances
IMPORTANCES = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)

# ─────────────────────────────────────────────
# 3.  HELPER: figure → base64 PNG  (for Gradio Image)
# ─────────────────────────────────────────────

import tempfile

def fig_to_file(fig):
    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".png"
    )

    fig.savefig(
        temp_file.name,
        format="png",
        dpi=110,
        bbox_inches="tight"
    )

    plt.close(fig)

    return temp_file.name


# ─────────────────────────────────────────────
# 4.  DASHBOARD CHARTS
# ─────────────────────────────────────────────

ACCENT   = "#3b82d4"
DANGER   = "#ef4444"
MUTED    = "#57606a"
SOFT_BG  = "#f7f8fa"

def chart_attrition_distribution():
    counts = df_raw["Attrition"].value_counts()
    fig, ax = plt.subplots(figsize=(5, 4), facecolor=SOFT_BG)
    bars = ax.bar(counts.index, counts.to_numpy(),
                  color=[ACCENT, DANGER], width=0.4, edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, counts.to_numpy()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(val), ha="center", va="bottom", fontsize=11, color=MUTED)
    ax.set_title("Attrition Distribution", fontsize=13, pad=10)
    ax.set_ylabel("Number of Employees")
    ax.set_facecolor(SOFT_BG)
    ax.spines[["top", "right"]].set_visible(False)
    pct = counts["Yes"] / counts.sum() * 100
    ax.set_xlabel(f"Overall attrition rate: {pct:.1f}%", color=MUTED, fontsize=10)
    fig.tight_layout()
    return fig_to_file(fig)


def chart_age_vs_attrition():
    fig, ax = plt.subplots(figsize=(6, 4), facecolor=SOFT_BG)
    stayed = df_raw[df_raw["Attrition"] == "No"]["Age"]
    left   = df_raw[df_raw["Attrition"] == "Yes"]["Age"]
    ax.hist(stayed, bins=20, alpha=0.7, color=ACCENT,  label="Stayed", edgecolor="white")
    ax.hist(left,   bins=20, alpha=0.7, color=DANGER, label="Left",   edgecolor="white")
    ax.set_title("Age Distribution by Attrition", fontsize=13, pad=10)
    ax.set_xlabel("Age"); ax.set_ylabel("Count")
    ax.set_facecolor(SOFT_BG)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend()
    fig.tight_layout()
    return fig_to_file(fig)


def chart_department_attrition():
    grp = df_raw.groupby("Department")["Attrition"].value_counts(normalize=True).unstack().fillna(0) * 100
    fig, ax = plt.subplots(figsize=(6, 4), facecolor=SOFT_BG)
    grp["Yes"].sort_values().plot(kind="barh", ax=ax, color=DANGER, edgecolor="white")
    ax.set_title("Attrition Rate by Department (%)", fontsize=13, pad=10)
    ax.set_xlabel("Attrition %")
    ax.set_facecolor(SOFT_BG)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig_to_file(fig)


def chart_income_attrition():
    fig, ax = plt.subplots(figsize=(6, 4), facecolor=SOFT_BG)
    stayed = df_raw[df_raw["Attrition"] == "No"]["MonthlyIncome"]
    left   = df_raw[df_raw["Attrition"] == "Yes"]["MonthlyIncome"]
    bp = ax.boxplot([stayed, left], patch_artist=True, widths=0.4,
                    medianprops=dict(color="white", linewidth=2))
    for patch, color in zip(bp["boxes"], [ACCENT, DANGER]):
        patch.set_facecolor(color); patch.set_alpha(0.8)
    ax.set_xticklabels(["Stayed", "Left"])
    ax.set_title("Monthly Income vs Attrition", fontsize=13, pad=10)
    ax.set_ylabel("Monthly Income ($)")
    ax.set_facecolor(SOFT_BG)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig_to_file(fig)


def chart_overtime_attrition():
    grp = df_raw.groupby("OverTime")["Attrition"].value_counts(normalize=True).unstack().fillna(0) * 100
    fig, ax = plt.subplots(figsize=(5, 4), facecolor=SOFT_BG)
    grp.plot(kind="bar", ax=ax, color=[ACCENT, DANGER], edgecolor="white")
    ax.set_title("OverTime vs Attrition Rate (%)", fontsize=13, pad=10)
    ax.set_xlabel("OverTime"); ax.set_ylabel("%"); ax.tick_params(axis="x", rotation=0)
    ax.set_facecolor(SOFT_BG)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(["Stayed", "Left"])
    fig.tight_layout()
    return fig_to_file(fig)


# ─────────────────────────────────────────────
# 5.  MODEL INSIGHT CHARTS
# ─────────────────────────────────────────────

def chart_feature_importance():
    top15 = IMPORTANCES.head(15)
    fig, ax = plt.subplots(figsize=(7, 5), facecolor=SOFT_BG)
    top15[::-1].plot(kind="barh", ax=ax, color=ACCENT, edgecolor="white")  # type: ignore[union-attr]
    ax.set_title("Top 15 Feature Importances (Random Forest)", fontsize=13, pad=10)
    ax.set_xlabel("Importance Score")
    ax.set_facecolor(SOFT_BG)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig_to_file(fig)


def chart_roc_curve():
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(5, 4), facecolor=SOFT_BG)
    ax.plot(fpr, tpr, color=ACCENT, lw=2, label=f"ROC AUC = {ROC_AUC:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color=MUTED)
    ax.set_title("ROC Curve", fontsize=13, pad=10)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.set_facecolor(SOFT_BG)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig_to_file(fig)


def chart_confusion_matrix():
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3.5), facecolor=SOFT_BG)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Stayed", "Left"]); ax.set_yticklabels(["Stayed", "Left"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix", fontsize=13, pad=10)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig_to_file(fig)


# ─────────────────────────────────────────────
# 6.  PREDICTION FUNCTION
# ─────────────────────────────────────────────

def predict_attrition(
    Age, BusinessTravel, DailyRate, Department, DistanceFromHome,
    Education, EducationField, EnvironmentSatisfaction, Gender,
    HourlyRate, JobInvolvement, JobLevel, JobRole, JobSatisfaction,
    MaritalStatus, MonthlyIncome, MonthlyRate, NumCompaniesWorked,
    OverTime, PercentSalaryHike, PerformanceRating, RelationshipSatisfaction,
    StockOptionLevel, TotalWorkingYears, TrainingTimesLastYear,
    WorkLifeBalance, YearsAtCompany, YearsInCurrentRole,
    YearsSinceLastPromotion, YearsWithCurrManager,
):
    raw = {
        "Age": Age,
        "BusinessTravel": BusinessTravel,
        "DailyRate": DailyRate,
        "Department": Department,
        "DistanceFromHome": DistanceFromHome,
        "Education": Education,
        "EducationField": EducationField,
        "EnvironmentSatisfaction": EnvironmentSatisfaction,
        "Gender": Gender,
        "HourlyRate": HourlyRate,
        "JobInvolvement": JobInvolvement,
        "JobLevel": JobLevel,
        "JobRole": JobRole,
        "JobSatisfaction": JobSatisfaction,
        "MaritalStatus": MaritalStatus,
        "MonthlyIncome": MonthlyIncome,
        "MonthlyRate": MonthlyRate,
        "NumCompaniesWorked": NumCompaniesWorked,
        "OverTime": OverTime,
        "PercentSalaryHike": PercentSalaryHike,
        "PerformanceRating": PerformanceRating,
        "RelationshipSatisfaction": RelationshipSatisfaction,
        "StockOptionLevel": StockOptionLevel,
        "TotalWorkingYears": TotalWorkingYears,
        "TrainingTimesLastYear": TrainingTimesLastYear,
        "WorkLifeBalance": WorkLifeBalance,
        "YearsAtCompany": YearsAtCompany,
        "YearsInCurrentRole": YearsInCurrentRole,
        "YearsSinceLastPromotion": YearsSinceLastPromotion,
        "YearsWithCurrManager": YearsWithCurrManager,
        "EmployeeNumber": 9999,
    }

    row = pd.DataFrame([raw])

    # Encode categoricals
    for col in CAT_COLS:
        le = ENCODERS[col]
        val = raw.get(col, "Unknown")
        if val not in le.classes_:
            val = le.classes_[0]
        row[col + "_enc"] = le.transform([str(val)])[0]

    input_vec = row.reindex(columns=FEATURE_COLS, fill_value=0)
    prob = model.predict_proba(input_vec)[0][1]
    pred = int(prob >= 0.5)

    # Risk band
    if prob < 0.30:
        band = "🟢  Low Risk"
        advice = "Employee appears stable. Maintain current engagement strategies."
    elif prob < 0.60:
        band = "🟡  Medium Risk"
        advice = "Some attrition signals detected. Consider a check-in or growth opportunities."
    else:
        band = "🔴  High Risk"
        advice = "High probability of attrition. Immediate retention action recommended."

    result = (
        f"### Prediction: {'⚠️ Will Leave' if pred else '✅ Will Stay'}\n\n"
        f"**Attrition Probability:** `{prob*100:.1f}%`\n\n"
        f"**Risk Band:** {band}\n\n"
        f"**Recommendation:** {advice}"
    )

    # Gauge chart
    fig, ax = plt.subplots(figsize=(4, 2.5), facecolor="white")
    color = DANGER if prob >= 0.60 else ("#f59e0b" if prob >= 0.30 else ACCENT)
    ax.barh([0], [prob],   color=color, height=0.4)
    ax.barh([0], [1 - prob], left=[prob], color="#e5e7eb", height=0.4)
    ax.set_xlim(0, 1); ax.set_ylim(-0.5, 0.5)
    ax.set_xticks([0, 0.25, 0.50, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_yticks([])
    ax.set_title(f"Attrition Probability: {prob*100:.1f}%", fontsize=12, pad=8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout()
    gauge_buf = fig_to_file(fig)

    return result, gauge_buf


# ─────────────────────────────────────────────
# 7.  UNIQUE VALUES FOR DROPDOWNS
# ─────────────────────────────────────────────

def uv(col):
    return sorted(df_raw[col].dropna().unique().tolist())


# ─────────────────────────────────────────────
# 8.  GRADIO UI
# ─────────────────────────────────────────────

THEME = gr_themes.Soft(
    primary_hue="blue",
    neutral_hue="slate",
    font=[gr_themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
)

with gr.Blocks(theme=THEME, title="Employee Attrition Predictor") as demo:

    gr.Markdown(
        """
        # 👥 Employee Attrition Prediction
        **Model:** Random Forest  |  **Dataset:** IBM HR Analytics  |  
        **Accuracy:** {:.1f}%  |  **ROC-AUC:** {:.3f}  |  **CV ROC-AUC:** {:.3f} ± {:.3f}
        """.format(ACCURACY * 100, ROC_AUC, CV_SCORES.mean(), CV_SCORES.std())
    )

    with gr.Tabs():

        # ── TAB 1: DASHBOARD ──────────────────────────────────────────
        with gr.TabItem("📊 Dashboard"):
            gr.Markdown("### Dataset Overview")
            with gr.Row():
                gr.Markdown(f"""
                | Metric | Value |
                |--------|-------|
                | Total Employees | {len(df_raw):,} |
                | Features Used | {len(FEATURE_COLS)} |
                | Attrition (Yes) | {(df_raw['Attrition']=='Yes').sum()} ({(df_raw['Attrition']=='Yes').mean()*100:.1f}%) |
                | Attrition (No) | {(df_raw['Attrition']=='No').sum()} ({(df_raw['Attrition']=='No').mean()*100:.1f}%) |
                | Avg Age | {df_raw['Age'].mean():.1f} |
                | Avg Monthly Income | ${df_raw['MonthlyIncome'].mean():,.0f} |
                """)

            gr.Markdown("### Exploratory Charts")
            with gr.Row():
                img_dist = gr.Image(label="Attrition Distribution", type="filepath")
                img_age  = gr.Image(label="Age vs Attrition",       type="filepath")
            with gr.Row():
                img_dept = gr.Image(label="Department Attrition %", type="filepath")
                img_inc  = gr.Image(label="Income vs Attrition",    type="filepath")
            with gr.Row():
                img_ot   = gr.Image(label="OverTime vs Attrition",  type="filepath")

            load_btn = gr.Button("🔄 Load Charts", variant="primary")

            def load_dashboard():
                return (
                    chart_attrition_distribution(),
                    chart_age_vs_attrition(),
                    chart_department_attrition(),
                    chart_income_attrition(),
                    chart_overtime_attrition(),
                )

            load_btn.click(
                load_dashboard,
                inputs=[],
                outputs=[img_dist, img_age, img_dept, img_inc, img_ot],
            )

        # ── TAB 2: PREDICT ────────────────────────────────────────────
        with gr.TabItem("🔮 Predict Attrition"):
            gr.Markdown("### Enter Employee Details")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("**Personal Info**")
                    Age                   = gr.Slider(18, 65, value=35,  step=1,  label="Age")
                    Gender                = gr.Dropdown(uv("Gender"),       label="Gender",       value=uv("Gender")[0])
                    MaritalStatus         = gr.Dropdown(uv("MaritalStatus"),label="Marital Status",value=uv("MaritalStatus")[0])
                    EducationField        = gr.Dropdown(uv("EducationField"),label="Education Field",value=uv("EducationField")[0])
                    Education             = gr.Slider(1, 5, value=3, step=1, label="Education Level (1=Below College, 5=Doctor)")
                    DistanceFromHome      = gr.Slider(1, 30, value=5, step=1, label="Distance From Home (km)")

                with gr.Column():
                    gr.Markdown("**Job Details**")
                    Department            = gr.Dropdown(uv("Department"),   label="Department",    value=uv("Department")[0])
                    JobRole               = gr.Dropdown(uv("JobRole"),       label="Job Role",      value=uv("JobRole")[0])
                    JobLevel              = gr.Slider(1, 5, value=2, step=1, label="Job Level")
                    JobInvolvement        = gr.Slider(1, 4, value=3, step=1, label="Job Involvement (1–4)")
                    JobSatisfaction       = gr.Slider(1, 4, value=3, step=1, label="Job Satisfaction (1–4)")
                    BusinessTravel        = gr.Dropdown(uv("BusinessTravel"),label="Business Travel",value=uv("BusinessTravel")[0])
                    OverTime              = gr.Dropdown(["Yes", "No"],       label="Over Time",     value="No")

                with gr.Column():
                    gr.Markdown("**Compensation & Experience**")
                    MonthlyIncome         = gr.Slider(1000, 20000, value=5000, step=100, label="Monthly Income ($)")
                    DailyRate             = gr.Slider(100, 1500,  value=800,  step=10,  label="Daily Rate")
                    HourlyRate            = gr.Slider(30, 100,    value=65,   step=1,   label="Hourly Rate")
                    MonthlyRate           = gr.Slider(2000, 27000, value=14000,step=100, label="Monthly Rate")
                    PercentSalaryHike     = gr.Slider(11, 25,     value=15,   step=1,   label="% Salary Hike")
                    StockOptionLevel      = gr.Slider(0, 3,       value=1,    step=1,   label="Stock Option Level")
                    NumCompaniesWorked    = gr.Slider(0, 9,       value=2,    step=1,   label="# Companies Worked")
                    TotalWorkingYears     = gr.Slider(0, 40,      value=8,    step=1,   label="Total Working Years")

                with gr.Column():
                    gr.Markdown("**Environment & Growth**")
                    EnvironmentSatisfaction = gr.Slider(1, 4, value=3, step=1, label="Environment Satisfaction (1–4)")
                    RelationshipSatisfaction= gr.Slider(1, 4, value=3, step=1, label="Relationship Satisfaction (1–4)")
                    WorkLifeBalance         = gr.Slider(1, 4, value=3, step=1, label="Work-Life Balance (1–4)")
                    PerformanceRating       = gr.Slider(3, 4, value=3, step=1, label="Performance Rating (3–4)")
                    TrainingTimesLastYear   = gr.Slider(0, 6, value=2, step=1, label="Trainings Last Year")
                    YearsAtCompany          = gr.Slider(0, 40, value=5, step=1, label="Years at Company")
                    YearsInCurrentRole      = gr.Slider(0, 18, value=3, step=1, label="Years in Current Role")
                    YearsSinceLastPromotion = gr.Slider(0, 15, value=1, step=1, label="Years Since Last Promotion")
                    YearsWithCurrManager    = gr.Slider(0, 17, value=3, step=1, label="Years with Current Manager")

            predict_btn = gr.Button("🚀 Predict Attrition", variant="primary", size="lg")

            with gr.Row():
                result_md   = gr.Markdown(label="Prediction Result")
                gauge_img   = gr.Image(label="Probability Gauge", type="filepath", height=200)

            predict_btn.click(
                predict_attrition,
                inputs=[
                    Age, BusinessTravel, DailyRate, Department, DistanceFromHome,
                    Education, EducationField, EnvironmentSatisfaction, Gender,
                    HourlyRate, JobInvolvement, JobLevel, JobRole, JobSatisfaction,
                    MaritalStatus, MonthlyIncome, MonthlyRate, NumCompaniesWorked,
                    OverTime, PercentSalaryHike, PerformanceRating, RelationshipSatisfaction,
                    StockOptionLevel, TotalWorkingYears, TrainingTimesLastYear,
                    WorkLifeBalance, YearsAtCompany, YearsInCurrentRole,
                    YearsSinceLastPromotion, YearsWithCurrManager,
                ],
                outputs=[result_md, gauge_img],
            )

        # ── TAB 3: MODEL INSIGHTS ─────────────────────────────────────
        with gr.TabItem("📈 Model Insights"):
            gr.Markdown("### Model Performance")
            with gr.Row():
                gr.Markdown(f"""
                | Metric | Value |
                |--------|-------|
                | Test Accuracy | {ACCURACY*100:.2f}% |
                | ROC-AUC (Test) | {ROC_AUC:.4f} |
                | CV ROC-AUC (5-fold) | {CV_SCORES.mean():.4f} ± {CV_SCORES.std():.4f} |
                | Algorithm | Random Forest (200 trees) |
                | Class Weighting | Balanced |
                """)

            gr.Markdown("### Classification Report")
            gr.Textbox(value=str(CLF_REPORT), label="", lines=10, interactive=False)

            gr.Markdown("### Visual Analysis")
            with gr.Row():
                img_fi  = gr.Image(label="Feature Importance",  type="filepath")
                img_roc = gr.Image(label="ROC Curve",           type="filepath")
                img_cm  = gr.Image(label="Confusion Matrix",    type="filepath")

            insights_btn = gr.Button("🔄 Load Insights", variant="primary")

            def load_insights():
                return chart_feature_importance(), chart_roc_curve(), chart_confusion_matrix()

            insights_btn.click(
                load_insights, inputs=[], outputs=[img_fi, img_roc, img_cm]
            )

    gr.Markdown(
        "<div style='text-align:center;color:#57606a;font-size:12px;"
        "border-top:1px solid #e5e7eb;margin-top:24px;padding-top:10px'>"
        "Employee Attrition Predictor · Built with Python, scikit-learn &amp; Gradio"
        "</div>"
    )

# ─────────────────────────────────────────────
# 9.  ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    demo.launch(share=False, server_port=7860, inbrowser=True)


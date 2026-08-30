from shiny import App, ui, render, reactive, req
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. LOAD EXPLICIT PIPELINE ARTIFACTS
# ==========================================
artifacts = joblib.load('loan_decision_artifacts.pkl')
imputer = artifacts['imputer']
scaler = artifacts['scaler']
encoder = artifacts['encoder']
model = artifacts['model']

# ==========================================
# 2. DEFINE THE UI LAYOUT
# ==========================================
app_ui = ui.page_navbar(
    ui.nav_panel("Loan Predictor",
        ui.layout_sidebar(
            ui.sidebar(
                # Accordion keeps the long form scannable/collapsible instead of one long scroll.
                ui.accordion(
                    ui.accordion_panel(
                        "👤 Applicant Details",
                        ui.input_numeric("age", "Age", value=35, min=18, max=100),
                        # Option labels/values must match the exact categories the model was trained on.
                        ui.input_select("marital", "Marital Status", {"Single": "Single", "Married": "Married", "Divorced": "Divorced", "Widowed": "Widowed"}),
                        ui.input_select("edu", "Education Level", {"High School": "High School", "Associate": "Associate", "Bachelor": "Bachelor", "Master": "Master", "Doctorate": "Doctorate"}),
                        ui.input_select("emp", "Employment Status", {"Employed": "Employed", "Self-Employed": "Self-Employed", "Unemployed": "Unemployed"}),
                        ui.input_select("home", "Home Ownership", {"Rent": "Rent", "Mortgage": "Mortgage", "Own": "Own", "Other": "Other"}),
                        value="applicant",
                    ),
                    ui.accordion_panel(
                        "💰 Financial Profile",
                        ui.input_numeric("income", "Annual Income (Rs)", value=60000, min=0),
                        ui.input_numeric("networth", "Net Worth (Rs)", value=100000),
                        ui.input_numeric("monthly_debt", "Other Monthly Debt (Rs)", value=500, min=0),
                        ui.input_numeric("credit_score", "Credit Score (300-850)", value=700, min=300, max=850),
                        value="financial",
                    ),
                    ui.accordion_panel(
                        "🏦 Loan Request",
                        ui.input_numeric("loan_amt", "Loan Amount (Rs)", value=20000, min=0),
                        ui.input_numeric("loan_dur", "Duration (Months)", value=60, min=1),
                        ui.input_numeric("int_rate", "Expected Interest Rate (%)", value=5.5, min=0, step=0.1),
                        ui.input_select("loan_purpose", "Loan Purpose", {"Auto": "Auto", "Debt Consolidation": "Debt Consolidation", "Education": "Education", "Home": "Home", "Other": "Other"}),
                        value="loan",
                    ),
                    ui.accordion_panel(
                        "📋 Optional Financial History",
                        ui.tooltip(
                            ui.tags.span("ℹ️ Why leave these blank?", class_="text-muted", style="font-size: 0.85em; cursor: help;"),
                            "The AI estimates any of these you skip from the rest of the applicant's profile, so only fill in what you know.",
                        ),
                        ui.input_numeric("chk_bal", "Checking Account Balance (Rs)", value=None),
                        ui.input_numeric("savings_bal", "Savings Account Balance (Rs)", value=None),
                        ui.input_numeric("cc_util", "Credit Card Utilization Rate (%)", value=None, min=0, max=100),
                        ui.input_numeric("job_ten", "Job Tenure (Years)", value=None, min=0),
                        ui.input_numeric("cred_hist", "Length of Credit History (Years)", value=None, min=0),
                        ui.input_numeric("bankruptcies", "Bankruptcy History (Count)", value=None, min=0),
                        ui.input_numeric("risk_score", "Internal Risk Score", value=None),
                        ui.input_numeric("liabilities", "Total Liabilities (Rs)", value=None, min=0),
                        ui.input_numeric("dependents", "Number of Dependents", value=None, min=0),
                        ui.input_numeric("open_credit_lines", "Number of Open Credit Lines", value=None, min=0),
                        ui.input_numeric("credit_inquiries", "Number of Recent Credit Inquiries", value=None, min=0),
                        ui.input_numeric("payment_history", "Payment History Score", value=None, min=0),
                        ui.input_numeric("prev_defaults", "Previous Loan Defaults (0 = No, 1 = Yes)", value=None, min=0, max=1),
                        value="optional",
                    ),
                    open=["applicant", "financial", "loan"],
                ),
                ui.input_action_button("predict_btn", "🔍 Evaluate Application", class_="btn-primary w-100 mt-3"),
                width=380,
            ),
            ui.output_ui("decision_panel"),
        )
    ),
    
    ui.nav_panel("Model Improvements",
        ui.card(
            ui.h4("Why HistGradientBoostingClassifier Best Suits Loan Prediction"),
            ui.markdown("""
            * **Captures Non-Linear Risk Interactions:** Loan default risk depends on interacting factors (e.g. high income but poor credit history, or low DTI but a bankruptcy record). Gradient-boosted decision trees split on these interactions directly, whereas a linear model like Logistic Regression would miss them unless every interaction term was hand-engineered.
            * **Built for Tabular, Mixed-Type Financial Data:** The dataset mixes continuous numeric fields (income, credit score, DTI) with categorical fields (education, home ownership, loan purpose). Histogram-based boosting bins continuous features internally, giving it the speed of a simple model with the accuracy of a complex one on exactly this kind of tabular data.
            * **Resilient to Outliers and Skewed Distributions:** Financial fields like net worth or liabilities can have extreme outliers. Tree-based splits are based on relative ordering, not raw magnitude, so a few extreme applicants don't distort the decision boundary the way they would in a distance- or gradient-magnitude-sensitive model.
            * **Pairs Well with SMOTE Balancing:** Loan approval datasets are typically imbalanced (far fewer rejections/defaults than approvals). Balancing the classes with `SMOTE` before training lets the boosted trees learn a genuine decision boundary for the minority class instead of defaulting to the majority class, avoiding the Accuracy Paradox.
            * **Robust to Missing Data After Imputation:** Optional applicant fields (job tenure, checking balance, etc.) are frequently blank. `KNNImputer` fills these using the most similar applicants, and boosting is tolerant of the resulting estimated values since it splits on thresholds rather than relying on exact precision.
            * **Prevents Data Leakage by Design:** Imputation, scaling, and encoding are fit once on training data and reused as fixed artifacts at inference time, so the model's real-world accuracy reflects its true generalization performance rather than information leaked from the test set.
            """)
        )
    ),
    header=ui.tags.style("""
        body, .navbar, .nav-link, .btn, input, select, textarea,
        h1, h2, h3, h4, h5, h6, p, span, div, label {
            font-family: 'Times New Roman', Times, serif !important;
        }

        body {
            background: linear-gradient(135deg, #eef2f7 0%, #dbe4f0 100%) fixed;
            color: #1f2b3d;
            min-height: 100vh;
        }

        .navbar {
            background-color: #1f2b3d !important;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
        }
        .navbar .nav-link, .navbar-brand, .navbar .navbar-text {
            color: #f5f7fa !important;
        }
        .navbar .nav-link:hover {
            color: #9fc0ff !important;
        }

        [class*="sidebar"] {
            background-color: #f8f9fb !important;
            border-right: 1px solid #dde3ea;
        }

        .card {
            background-color: #ffffff;
            border: none;
            border-radius: 10px;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.1);
        }

        .accordion-button {
            background-color: #f1f4f9;
            color: #1f2b3d;
        }
        .accordion-button:not(.collapsed) {
            background-color: #e3ebfa;
            color: #1f2b3d;
        }

        .form-control, .form-select {
            background-color: #ffffff;
            border: 1px solid #cfd7e3;
            color: #1f2b3d;
        }

        .value-box {
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        }

        .table, .dataframe, .shiny-data-grid {
            background-color: #ffffff;
        }
    """),
    title="Intelligent Loan Decision Engine",
    fillable=True,
)

# ==========================================
# 3. DEFINE THE SERVER LOGIC
# ==========================================
def server(input, output, session):
    
    @reactive.Calc
    @reactive.event(input.predict_btn)
    def process_application():
        # Backend Financial Engineering
        monthly_income = input.income() / 12 if input.income() else 0
        r = (input.int_rate() / 100) / 12 if input.int_rate() else 0
        n = input.loan_dur() if input.loan_dur() else 1
        P = input.loan_amt() if input.loan_amt() else 0
        m_debt = input.monthly_debt() if input.monthly_debt() else 0
        
        if r > 0:
            monthly_loan_pmt = P * r * ((1 + r)**n) / (((1 + r)**n) - 1)
        else:
            monthly_loan_pmt = P / n
            
        dti = ((monthly_loan_pmt + m_debt) * 100) / monthly_income if monthly_income > 0 else 0
        
        def handle_missing(val):
            return np.nan if val is None or val == "" else val

        # 1. Capture known UI values (keys must match the artifact's trained feature names exactly)
        user_inputs = {
            'Age': handle_missing(input.age()),
            'LoanAmount': P,
            'LoanDuration': n,
            'InterestRate': handle_missing(input.int_rate()),
            'MonthlyDebtPayments': m_debt,
            'CreditScore': handle_missing(input.credit_score()),
            'RiskScore': handle_missing(input.risk_score()),
            'NetWorth': handle_missing(input.networth()),
            'MaritalStatus': input.marital(),
            'EducationLevel': input.edu(),
            'EmploymentStatus': input.emp(),
            'HomeOwnershipStatus': input.home(),
            'LoanPurpose': input.loan_purpose(),
            'Monthlyincome': monthly_income,
            'Monthlyloanpyament': monthly_loan_pmt,
            'Totaldebttoincomeratio': dti,
            'CheckingAccountBalance': handle_missing(input.chk_bal()),
            'SavingsAccountBalance': handle_missing(input.savings_bal()),
            'CreditCardUtilizationRate': handle_missing(input.cc_util()),
            'JobTenure': handle_missing(input.job_ten()),
            'LengthOfCreditHistory': handle_missing(input.cred_hist()),
            'BankruptcyHistory': handle_missing(input.bankruptcies()),
            'TotalLiabilities': handle_missing(input.liabilities()),
            'NumberOfDependents': handle_missing(input.dependents()),
            'NumberOfOpenCreditLines': handle_missing(input.open_credit_lines()),
            'NumberOfCreditInquiries': handle_missing(input.credit_inquiries()),
            'PaymentHistory': handle_missing(input.payment_history()),
            'PreviousLoanDefaults': handle_missing(input.prev_defaults()),
        }
        
        # 2. Extract expected columns from artifacts to guarantee exact alignment
        expected_num_cols = list(imputer.feature_names_in_)
        expected_cat_cols = list(encoder.feature_names_in_)
        
        # 3. Build dictionary ensuring ALL expected columns are present
        full_input_dict = {}
        for col in expected_num_cols:
            full_input_dict[col] = user_inputs.get(col, np.nan) 
            
        for col in expected_cat_cols:
            full_input_dict[col] = user_inputs.get(col, 'Unknown') 

        input_data = pd.DataFrame([full_input_dict])
        
        # 4. Explicit Preprocessing Steps
        num_imputed = imputer.transform(input_data[expected_num_cols])
        num_scaled = scaler.transform(num_imputed)
        cat_encoded = encoder.transform(input_data[expected_cat_cols])
        
        X_processed = np.hstack((num_scaled, cat_encoded))
        
        # predict_proba(...)[:, 1] is the probability of class 1 (LoanApproved == 1, i.e. "Yes")
        approval_prob = model.predict_proba(X_processed)[0][1] * 100
        prediction = model.predict(X_processed)[0]
        
        return approval_prob, prediction, monthly_income, monthly_loan_pmt, dti

    @render.ui
    def decision_panel():
        if input.predict_btn() == 0:
            return ui.card(
                ui.h3("Decision Output"),
                ui.p(
                    "Fill in the applicant's details on the left and click ",
                    ui.tags.b("Evaluate Application"),
                    " to get an instant, AI-backed loan decision.",
                    class_="text-muted",
                ),
            )

        try:
            approval_prob, pred, _, _, _ = process_application()
        except Exception as e:
            ui.notification_show(f"Could not evaluate this application: {e}", type="error", duration=8)
            return ui.card(
                ui.h3("Decision Output"),
                ui.p(f"⚠️ Unable to evaluate application: {e}", style="color: red;"),
            )

        approved = pred == 1
        decision_label = "Approved" if approved else "Denied"
        decision_theme = "success" if approved else "danger"

        if approval_prob < 30:
            risk_label, risk_theme = "🔴 High Risk", "danger"
        elif approval_prob < 60:
            risk_label, risk_theme = "🟠 Moderate Risk", "warning"
        else:
            risk_label, risk_theme = "🟢 Low Risk", "success"

        return ui.TagList(
            ui.layout_columns(
                ui.value_box(
                    "Decision",
                    decision_label,
                    showcase=ui.tags.span("✅" if approved else "❌", style="font-size: 2rem;"),
                    theme=decision_theme,
                ),
                ui.value_box(
                    "Approval Probability",
                    f"{approval_prob:.1f}%",
                    theme="primary",
                ),
                ui.value_box(
                    "Risk Level",
                    risk_label,
                    theme=risk_theme,
                ),
                col_widths=[4, 4, 4],
            ),
            ui.card(
                ui.card_header("Calculated Applicant Financial Ratios"),
                ui.output_data_frame("ratios_table"),
            ),
        )

    @render.data_frame
    def ratios_table():
        req(input.predict_btn() > 0)
        _, _, m_inc, m_pmt, dti = process_application()
        df = pd.DataFrame({
            "Financial Metric": ["Est. Monthly Income", "Est. New Loan Payment", "Total Debt-to-Income Ratio"],
            "Calculated Value": [f"Rs{m_inc:,.2f}", f"Rs{m_pmt:,.2f}", f"{dti:.1f}%"]
        })
        return render.DataGrid(df, width="100%")

app = App(app_ui, server)

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from llm import talk
from fastapi.middleware.cors import CORSMiddleware

model = joblib.load("loan_ensemble_model.pkl")

occupation_map = {"Labour": 0, "Farmer": 1, "Trader": 2, "Gig Worker": 3, "Student": 4}
age_group_map = {"18-25": 0, "26-35": 1, "36-45": 2, "46-60": 3, "60+": 4}
education_map = {"Illiterate": 0, "Primary": 1, "10th": 2, "Inter": 3, "Graduate": 4}
district_map = {
    "Adilabad": 0,
    "Nizamabad": 1,
    "Karimnagar": 2,
    "Nalgonda": 3,
    "Warangal": 4,
    "Mahbubnagar": 5,
    "Rangareddy": 6,
    "Hyderabad": 7,
    "Khammam": 8,
    "Medak": 9,
}


def calculate_repayment_score(data):
    score = 0

    occupation_scores = [30, 45, 50, 40, 25]
    score += occupation_scores[occupation_map[data.occupation_type]]

    age_scores = [20, 25, 15, 10, 5]
    score += age_scores[age_group_map[data.age_group]]

    education_scores = [10, 15, 20, 20, 25]
    score += education_scores[education_map[data.education_level]]

    if data.monthly_income >= 15000:
        score += 60
    elif data.monthly_income >= 10000:
        score += 50
    elif data.monthly_income >= 5000:
        score += 40
    else:
        score += 25

    if data.household_dependents == 0:
        score += 25
    elif data.household_dependents <= 2:
        score += 20
    elif data.household_dependents <= 4:
        score += 15
    else:
        score += 10

    if data.bank_avg_monthly_balance >= 10000:
        score += 60
    elif data.bank_avg_monthly_balance >= 5000:
        score += 50
    elif data.bank_avg_monthly_balance >= 2000:
        score += 35
    else:
        score += 20

    if data.bank_credit_txn_count >= 20:
        score += 50
    elif data.bank_credit_txn_count >= 10:
        score += 40
    elif data.bank_credit_txn_count >= 5:
        score += 30
    else:
        score += 15

    if data.regional_credit_risk < 0.3:
        score += 30
    elif data.regional_credit_risk < 0.6:
        score += 20
    else:
        score += 10

    if data.nightlight_intensity >= 0.7:
        score += 20
    elif data.nightlight_intensity >= 0.4:
        score += 15
    else:
        score += 10

    if data.district_literacy_rate >= 75:
        score += 15
    elif data.district_literacy_rate >= 60:
        score += 10
    else:
        score += 5

    return score


class LoanApplication(BaseModel):
    occupation_type: str
    age_group: str
    education_level: str
    monthly_income: float
    household_dependents: int
    district: str
    regional_credit_risk: float
    nightlight_intensity: float
    district_literacy_rate: float
    bank_avg_monthly_balance: float
    bank_credit_txn_count: int
    bank_statement: str


app = FastAPI(
    title="Loan Approval Prediction API",
    version="1.2",
    description="Predicts loan approval and uses LLM to analyze the bank statement.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict")
def predict_loan(data: LoanApplication):
    input_data = {
        "occupation_type": occupation_map[data.occupation_type],
        "age_group": age_group_map[data.age_group],
        "education_level": education_map[data.education_level],
        "monthly_income": data.monthly_income,
        "household_dependents": data.household_dependents,
        "district": district_map[data.district],
        "regional_credit_risk": data.regional_credit_risk,
        "nightlight_intensity": data.nightlight_intensity,
        "district_literacy_rate": data.district_literacy_rate,
        "bank_avg_monthly_balance": data.bank_avg_monthly_balance,
        "bank_credit_txn_count": data.bank_credit_txn_count,
    }

    income_per_dependent = data.monthly_income / (data.household_dependents + 1e-5)
    balance_to_income_ratio = data.bank_avg_monthly_balance / (
        data.monthly_income + 1e-5
    )
    credit_txn_ratio = data.bank_credit_txn_count / (data.monthly_income + 1e-5)
    is_financially_strong = int(
        balance_to_income_ratio > 1 and data.bank_credit_txn_count > 5
    )

    input_df = pd.DataFrame(
        [
            {
                **input_data,
                "income_per_dependent": income_per_dependent,
                "balance_to_income_ratio": balance_to_income_ratio,
                "credit_txn_ratio": credit_txn_ratio,
                "is_financially_strong": is_financially_strong,
            }
        ]
    )

    prediction = model.predict(input_df)[0]
    prediction = "✅ Loan Approved" if prediction == 1 else "❌ Loan Not Approved"

    llm_analysis = talk(f"{prediction} {data.bank_statement}")

    repayment_score = calculate_repayment_score(data)

    return {
        "status": prediction,
        "repayment_score_out_of_500": repayment_score,
        "llm_analysis": llm_analysis,
    }

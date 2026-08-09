import pandas as pd
import joblib
import gradio as gr


# Load trained model
model = joblib.load("loan_approval_model.pkl")


def predict_loan(
    gender,
    married,
    dependents,
    education,
    self_employed,
    applicant_income,
    coapplicant_income,
    loan_amount,
    loan_term,
    credit_history,
    property_area
):

    total_income = applicant_income + coapplicant_income

    applicant = pd.DataFrame({
        "Gender": [gender],
        "Married": [married],
        "Dependents": [dependents],
        "Education": [education],
        "Self_Employed": [self_employed],
        "ApplicantIncome": [applicant_income],
        "CoapplicantIncome": [coapplicant_income],
        "LoanAmount": [loan_amount],
        "Loan_Amount_Term": [loan_term],
        "Credit_History": [credit_history],
        "Property_Area": [property_area],
        "TotalIncome": [total_income]
    })

    prediction = model.predict(applicant)[0]

    probability = model.predict_proba(applicant)[0][1]

    if prediction == 1:
        result = "🟢 LOAN APPROVED"
    else:
        result = "🔴 LOAN REJECTED"

    return f"{result}\nApproval probability: {probability:.2%}"


# Gradio interface
app = gr.Interface(
    fn=predict_loan,

    inputs=[
        gr.Dropdown(
            ["Male", "Female"],
            label="Gender"
        ),

        gr.Dropdown(
            ["Yes", "No"],
            label="Married"
        ),

        gr.Dropdown(
            ["0", "1", "2", "3+"],
            label="Dependents"
        ),

        gr.Dropdown(
            ["Graduate", "Not Graduate"],
            label="Education"
        ),

        gr.Dropdown(
            ["Yes", "No"],
            label="Self Employed"
        ),

        gr.Number(
            label="Applicant Income"
        ),

        gr.Number(
            label="Co-applicant Income"
        ),

        gr.Number(
            label="Loan Amount"
        ),

        gr.Number(
            label="Loan Amount Term"
        ),

        gr.Dropdown(
            [0, 1],
            label="Credit History"
        ),

        gr.Dropdown(
            ["Urban", "Semiurban", "Rural"],
            label="Property Area"
        )
    ],

    outputs=gr.Textbox(
        label="Prediction"
    ),

    title="🏠 Smart Loan Approval Predictor",

    description=(
        "Enter applicant financial and demographic "
        "details to predict loan approval."
    )
)


if __name__ == "__main__":
    app.launch()

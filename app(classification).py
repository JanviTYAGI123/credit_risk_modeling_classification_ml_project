Hugging Face's logo
Hugging Face
Models
Datasets
Spaces
Community
Docs
Pricing



Spaces:

abhinand462
/
credit_risk


like
0

App
Files
Community
Settings
credit_risk
/
app.py

abhinand462's picture
abhinand462
Update app.py
51bc4d5
verified
2 minutes ago
raw

Copy download link
history
blame
edit
delete

2.78 kB
import gradio as gr
import pandas as pd
import joblib

# Load your trained model
model = joblib.load('credit_risk_model.joblib')

# This is the exact list of columns your model was trained on.
MODEL_COLUMNS = [
    'age', 'loan_tenure_months', 'number_of_open_accounts', 
    'credit_utilization_ratio', 'loan_to_income', 'delinquency_ratio', 
    'avg_dpd_per_delinquency', 'residence_type_Owned', 
    'residence_type_Rented', 'loan_purpose_Education', 
    'loan_purpose_Home', 'loan_purpose_Personal', 'loan_type_Unsecured'
]

# Define the prediction function
def predict_credit_risk(age, residence_type, loan_purpose, loan_type, loan_tenure_months, number_of_open_accounts, credit_utilization_ratio, loan_to_income, delinquency_ratio, avg_dpd_per_delinquency):
    
    # Create a dictionary for the user's input
    input_dict = {
        'age': age,
        'loan_tenure_months': loan_tenure_months,
        'number_of_open_accounts': number_of_open_accounts,
        'credit_utilization_ratio': credit_utilization_ratio,
        'loan_to_income': loan_to_income,
        'delinquency_ratio': delinquency_ratio,
        'avg_dpd_per_delinquency': avg_dpd_per_delinquency,
        f'residence_type_{residence_type}': 1,
        f'loan_purpose_{loan_purpose}': 1,
        f'loan_type_{loan_type}': 1
    }

    # Convert to a DataFrame
    input_data = pd.DataFrame([input_dict])
    
    # Reindex the DataFrame to match the model's expected columns and order.
    # This is the crucial step that fixes the error.
    input_data = input_data.reindex(columns=MODEL_COLUMNS, fill_value=0)

    # Make the prediction
    prediction = model.predict(input_data)[0]
    
    if prediction == 1:
        return "High Risk"
    else:
        return "Low Risk"

# Create the Gradio web interface
# Note: The dropdowns/radios might contain options your model wasn't trained on (like 'Mortgage'),
# but the reindex step above will handle them correctly.
iface = gr.Interface(
    fn=predict_credit_risk,
    inputs=[
        gr.Number(label="Age"),
        gr.Radio(['Owned', 'Rented', 'Mortgage'], label="Residence Type"),
        gr.Dropdown(['Home', 'Personal', 'Education', 'Auto', 'Business'], label="Loan Purpose"),
        gr.Radio(['Secured', 'Unsecured'], label="Loan Type"),
        gr.Number(label="Loan Tenure (Months)"),
        gr.Number(label="Number of Open Accounts"),
        gr.Slider(0, 100, label="Credit Utilization Ratio (%)"),
        gr.Number(label="Loan to Income Ratio"),
        gr.Slider(0, 100, label="Delinquency Ratio (%)"),
        gr.Number(label="Average DPD per Delinquency")
    ],
    outputs="text",
    title="Credit Risk Prediction",
    description="Enter the applicant's details to predict their credit risk."
)

# Launch the app
iface.launch()

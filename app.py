#### IMPORTING LIBRARIES, MODELS AND DATA
import streamlit as st
import pandas as pd
import joblib

# Importing Data

# importing ML models and data
classificator = joblib.load('models/Classification/model_99precision.pkl')
classificator_scaler = joblib.load('models/Classification/model_99precision_scaler.pkl')

regressor = joblib.load('models/Regression/model_94RMSE.pkl')
regressor_scaler = joblib.load('models/Regression/model_94RMSE_scaler.pkl')

validation_data = pd.read_csv('data/val_dataset.csv')
# Lista con el nuevo orden de las columnas 
column_order = ['Age', 'AnnualIncome', 'CreditScore', 'EmploymentStatus', 'EducationLevel', 'Experience', 'LoanAmount', 'LoanDuration', 'MaritalStatus', 'NumberOfDependents', 'HomeOwnershipStatus', 'MonthlyDebtPayments', 'CreditCardUtilizationRate', 'NumberOfOpenCreditLines', 'NumberOfCreditInquiries', 'DebtToIncomeRatio', 'BankruptcyHistory', 'LoanPurpose', 'PreviousLoanDefaults', 'PaymentHistory', 'LengthOfCreditHistory', 'SavingsAccountBalance', 'CheckingAccountBalance', 'TotalAssets', 'TotalLiabilities', 'MonthlyIncome', 'UtilityBillsPaymentHistory', 'JobTenure', 'NetWorth', 'BaseInterestRate', 'InterestRate', 'MonthlyLoanPayment', 'TotalDebtToIncomeRatio', 'day','month','year'] # Reordenar las columnas del DataFrame
validation_data = validation_data.drop(columns=['Unnamed: 0','LoanApproved','RiskScore'])
validation_data.index.name = 'Loan ID'
validation_data = validation_data[column_order]

encoders_labels = ['EducationLevel','EmploymentStatus','HomeOwnershipStatus','LoanPurpose','MaritalStatus']
encoders = {encoder:joblib.load(f'models/encoder_{encoder}.pkl') for encoder in encoders_labels}

# PAGE STRUCTURE
st.title('APPLICATION FOR CLASSIFY LOANS AND CALCULATE THEIR RISK')
# Data entry
st.header('DATA ENTRY FOR A POSSIBLE LOAN OR CLIENT')
st.markdown('''
    This is an project to classify and calculate the risk of a possible or existent loan,
    in this case I use two ML models from skit-learn to calculate them, a linear regressor
    for calculate the risk, and a logistic regressor to classify the loans in two categories: 

    APPROVED OR NOT

    Trying to show a very simple and minimalistic interface        
    ''')

# dataframes
input_data = {}

show_validation_data = validation_data.copy()
show_validation_data['date'] = pd.to_datetime(show_validation_data['year'].astype(str) + '/' +
                                              show_validation_data['month'].astype(str) + '/' +
                                              show_validation_data['day'].astype(str)).dt.date

# Transformación inversa de las columnas categóricas 
for col in encoders_labels: 
    if col in show_validation_data.columns: 
        show_validation_data[col] = encoders[col].inverse_transform(show_validation_data[col])

# expander of validation data
with st.expander('Data in validation database'):
    st.dataframe(show_validation_data.drop(columns=['year','month','day']))
    row = st.number_input('Select Row Number of Table',0, len(validation_data))
    if st.button('Select'):
        st.session_state.data_entry_df =  validation_data.loc[[row],:]
        st.rerun()

# Crear un formulario en un diálogo
@st.dialog('Formulary input')
def data_entry():

    st.title('Entry Data')
    application_date = st.date_input('Application Date')
    input_data['Age'] = st.number_input('Age', 0, 130)
    input_data['AnnualIncome'] = st.number_input('Annual Income')
    input_data['CreditScore'] = st.number_input('Credit Score', 0, 1000)
    input_data['EmploymentStatus'] = st.selectbox('Employment Status', ['Employed', 'Self-Employed', 'Unemployed'])
    input_data['EducationLevel'] = st.selectbox('Education Level', ['High School', 'Bachelor', 'Associate', 'Master', 'Doctorate'])
    input_data['Experience'] = st.number_input('Experience', 0, 130)
    input_data['LoanAmount'] = st.number_input('Loan Amount', 0)
    input_data['LoanDuration'] = st.number_input('Loan Duration', 0)
    input_data['MaritalStatus'] = st.selectbox('Marital Status', ['Single', 'Married', 'Divorced', 'Widowed'])
    input_data['NumberOfDependents'] = st.number_input('Dependents', 0)
    input_data['HomeOwnershipStatus'] = st.selectbox('Homeowner Status', ['Own', 'Mortgage', 'Rent', 'Other'])
    input_data['MonthlyDebtPayments'] = st.number_input('Monthy Payments', 0)
    input_data['CreditCardUtilizationRate'] = st.number_input('Credit Card Utilization Rate', 0, 1)
    input_data['NumberOfOpenCreditLines'] = st.number_input('Number of open credit lines', 0)
    input_data['NumberOfCreditInquiries'] = st.number_input('Number of credit inquiries', 0)
    input_data['DebtToIncomeRatio'] = st.slider('Debt to Income Ratio', 0.0, 1.0, step=0.001)
    input_data['BankruptcyHistory'] = st.number_input('Bankruptcy History')
    input_data['LoanPurpose'] = st.selectbox('Loan Purpose', ['Home', 'Debt Consolidation', 'Education', 'Other', 'Auto'])
    loan_default = st.checkbox('Previous Loans Defaults')
    input_data['PreviousLoanDefaults'] = (1 if loan_default else 0)
    input_data['PaymentHistory'] = st.number_input('Payment History')
    input_data['LengthOfCreditHistory'] = st.number_input('Credit History Lenght', 0) #### Me he quedado aquí
    input_data['SavingsAccountBalance'] = st.number_input('Savings Account Balance')
    input_data['CheckingAccountBalance'] = st.number_input('Checking Account Balance')
    input_data['TotalAssets'] = st.number_input('Total Assets', 0)
    input_data['TotalLiabilities'] = st.number_input('Total Liabilities', 0)
    input_data['MonthlyIncome'] = st.number_input('Monthy Income', 0)
    input_data['UtilityBillsPaymentHistory'] = st.number_input('Utility Bills Payment History', 0, 1)
    input_data['JobTenure'] = st.number_input('Job Tenure', 0, 40)
    input_data['NetWorth'] = st.number_input('Net Worth', 0)
    input_data['BaseInterestRate'] = st.number_input('Base Interest Rate', 0, 1)
    input_data['InterestRate'] = st.number_input('Interest Rate', 0, 1)
    input_data['MonthlyLoanPayment'] = st.number_input('Monthly Loan Payment', 0)
    input_data['TotalDebtToIncomeRatio'] = st.number_input('Total Debt to Income Ratio', 0)
    input_data['day'] = application_date.day
    input_data['month'] = application_date.month
    input_data['year'] = application_date.year
    if st.button('Submit'):
        st.session_state.data_entry_df = pd.DataFrame([input_data])
        st.rerun()


if 'data_entry_df' not in st.session_state:
    st.write('Input data for a new loan')
    if st.button('Form'):
        data_entry()
else:
    # st.write(st.session_state.data_entry_df)
    df_input = st.session_state.data_entry_df
    
col1, col2 = st.columns(2)
# col1.
# Classification of the loan
col1.subheader('Approvation')
col2.subheader('Risk')

try:
    # transforming columns
    for col in df_input.select_dtypes('object').columns:
        df_input[col] = encoders[col].transform(df_input[col])
    # input for classificator
    input_classificator = classificator_scaler.transform(df_input.drop(columns=['EmploymentStatus', 'MaritalStatus', 'NumberOfDependents', 'CreditCardUtilizationRate','DebtToIncomeRatio', 'CheckingAccountBalance', 'JobTenure']))
    # input for regressor
    input_regressor = regressor_scaler.transform(df_input.drop(columns=['LoanDuration', 'NumberOfDependents', 'HomeOwnershipStatus','NumberOfOpenCreditLines', 'NumberOfCreditInquiries', 'LoanPurpose', 'PaymentHistory', 'SavingsAccountBalance', 'TotalLiabilities', 'day', 'month', 'year']))
    # predictions
    classification = ('Approved' if classificator.predict(input_classificator) == 1 else 'Not Approved')
    risk = round(regressor.predict(input_regressor)[0])
    # columns visualization
    st.success('Success Calculus', icon="✅") #success mesage

    col1.metric(label='Status', value = classification)
    col2.metric(label='Risk Value',value = risk)

except:
    st.warning('No Info of Invalid Info taken')
    col1.metric(label='Status', value = 'No data yet')
    col2.metric(label='Risk Value', value = 'NA')
    
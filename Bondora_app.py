import streamlit as st
import pandas as pd
import numpy as np
import warnings
import base64
import pickle

from PIL import Image
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Ridge

warnings.filterwarnings("ignore")
st.write("""
# Bondora Prediction
### Good luck!
""")
Data = pd.read_csv("FDD.csv")


def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
            background-size: cover
        }}
        </style>
        """,
        unsafe_allow_html=True
    )



im = Image.open("loan.jpg")
st.sidebar.image(im)
st.sidebar.header("Input data")
st.sidebar.write("*Kindly answer the following questions to check your **loan status**:*")


def InputData():
    BidsPortfolioManager=st.sidebar.number_input("Enter The amount of investment offers made by Portfolio Managers:")
    BidsApi=st.sidebar.number_input("Enter The amount of investment offers made via Api:")
    BidsManual=st.sidebar.number_input("Enter The amount of investment offers made manually:")
    NewCreditCustomer = st.sidebar.selectbox("Did you have prior credit history in Bondora (at least 3 months):",
                                   ['Yes','NO'])
    if (NewCreditCustomer == "Yes"):
       NewCreditCustomer = 'Existing_credit_customer'
    else:
       NewCreditCustomer = 'NewCreditCustomer'
    Age= st.sidebar.number_input("Enter your age:")
    AppliedAmount=st.sidebar.number_input("Enter The amount applied for originally:")
    Interest=st.sidebar.number_input("Enter The Maximum interest rate accepted in the loan application:")
    MonthlyPayment=st.sidebar.number_input("Enter The Estimated amount has to pay every month:")
    IncomeTotal=st.sidebar.number_input("Enter The total income:")
    ExistingLiabilities=st.sidebar.number_input("Enter The number of existing liabilities:")
    RefinanceLiabilities=st.sidebar.number_input("Enter The total amount of liabilities after refinancing:")
    DebtToIncome=st.sidebar.number_input("Enter The Ratio of monthly gross income that goes toward paying loans:")
    FreeCash=st.sidebar.number_input("Enter The Discretionary income after monthly liabilities:")
    Restructured=st.sidebar.selectbox("IS the original maturity date of the loan has been increased by more than 60 days:", ['Yes','No'])
    PrincipalPaymentsMade=st.sidebar.number_input("Enter Note owner received loan transfers principal amount:")
    InterestAndPenaltyPaymentsMade=st.sidebar.number_input("Enter Note owner received loan transfers earned interest, penalties total amount:")
    PreviousEarlyRepaymentsCountBeforeLoan=st.sidebar.number_input("Enter How many times the borrower had repaid early:")
    VerificationType=st.sidebar.selectbox("Enter the Method used for loan application data verification:",['Income_verified','Income_expenses_verified','Income_unverified','Income_unverified_crossref_phone','Not_set'])


    LanguageCode = st.sidebar.selectbox("Select language:",
                                   ["English", "Estonian", "Finnish", "German", "Russian", "Slovakian", "Spanish",
                                    "Other"])
    Gender = st.sidebar.selectbox("Select Gender:",
                                        ['Male','Woman','Undefined'])
    Country= st.sidebar.selectbox("Select Country:", ["EE", "ES", "FI", "SK"])
    UseOfLoan= st.sidebar.selectbox("Select Use Of Loan:", ["Home_improvement", "Loan_consolidation", "Vehicle", "Business",'Travel','Health','Education','Real_estate','Purchase_of_machinery_equipment','Other_business','Accounts_receivable_financing','Working_capital_financing','Acquisition_of_stocks','Acquisition_of_real_estate','Construction_finance','Not_set','Other'])
    Education=st.sidebar.selectbox("Select the Education:",
                                        ['Basic education','Primary education','Secondary education','Vocational education','Higher education'])
    MaritalStatus = st.sidebar.selectbox("Select the Marital Status:",
                                     ['Single', 'Married', 'Cohabitant',
                                      'Divorced', 'Widow','Not_specified'])
    EmploymentStatus = st.sidebar.selectbox("Select the Employment Status:",
                                         ['Fully', 'Entrepreneur', 'Retiree',
                                          'Self_employed', 'Partially', 'Not_specified'])
    EmploymentDurationCurrentEmployer = st.sidebar.selectbox("Select the Employment time with the current employer:",
                                            ['UpTo1Year', 'UpTo2Years', 'UpTo3Years',
                                             'UpTo4Years', 'UpTo5Years', 'MoreThan5Years','Retiree','TrialPeriod','Other'])
    OccupationArea = st.sidebar.selectbox("Select the Occupation Area:",
                                            ['Retail_and_wholesale', 'Construction', 'Processing',
                                             'Transport_and_warehousing', 'Healthcare_and_social_help', 'Hospitality_and_catering', 'Info_and_telecom', 'Civil_service_and_military',
                                             'Education','Finance_and_insurance','Agriculture_forestry_and_fishing','Administrative','Energy','Art_and_entertainment','Research','Real_estate','Utilities','Mining','Not_specified','Other'])
    HomeOwnershipType = st.sidebar.selectbox("Select the HomeOwnership Type:",
                                            ['Owner', 'Tenant_pre_furnished_property', 'Living_with_parents',
                                             'Mortgage', 'Tenant_unfurnished_property', 'Joint_ownership', 'Joint_tenant', 'Council_house',
                                              'Owner_with_encumbrance','Homeless','Not_specified','Other'])
    Rating = st.sidebar.selectbox("Enter the rating issued by the rating Model:",
                                  ["A", "AA", "B", "C", "D", "E", "F", "HR", "unkown"])
    CreditScoreEsMicroL = st.sidebar.selectbox("Enter the Credit Score EsMicroL:",
                                  ["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",'M10'])
    if (CreditScoreEsMicroL == "M1"):
       CreditScoreEsMicroL = 100
    elif (CreditScoreEsMicroL == "M2"):
       CreditScoreEsMicroL = 90
    elif (CreditScoreEsMicroL == "M3"):
       CreditScoreEsMicroL = 80
    elif (CreditScoreEsMicroL == "M4"):
       CreditScoreEsMicroL = 70
    elif (CreditScoreEsMicroL == "M5"):
       CreditScoreEsMicroL = 60
    elif (CreditScoreEsMicroL == "M6"):
       CreditScoreEsMicroL = 50
    elif (CreditScoreEsMicroL == "M7"):
       CreditScoreEsMicroL = 40
    elif (CreditScoreEsMicroL == "M8"):
       CreditScoreEsMicroL = 30
    elif (CreditScoreEsMicroL == "M9"):
       CreditScoreEsMicroL = 20
    elif (CreditScoreEsMicroL == "M10"):
       CreditScoreEsMicroL = 10
    else:
       CreditScoreEsMicroL = 0

    InterestAndPenaltyBalance = st.sidebar.number_input("Enter InterestAndPenaltyBalance:")
    PrincipalBalance = st.sidebar.number_input("Enter PrincipalBalance:")


    data = {"BidsPortfolioManager": BidsPortfolioManager,
            "BidsApi": BidsApi,
            "BidsManual": BidsManual,

            "NewCreditCustomer": NewCreditCustomer,
            "Age": Age,
            "AppliedAmount": AppliedAmount,
            "Interest": Interest,

            "MonthlyPayment": MonthlyPayment,
            "IncomeTotal": IncomeTotal,
            "ExistingLiabilities": ExistingLiabilities,
            'RefinanceLiabilities':RefinanceLiabilities,
            'DebtToIncome':DebtToIncome,
            'FreeCash':FreeCash,
            'Restructured':Restructured,
            'PrincipalPaymentsMade':PrincipalPaymentsMade,
            'InterestAndPenaltyPaymentsMade':InterestAndPenaltyPaymentsMade,
            'PreviousEarlyRepaymentsCountBeforeLoan':PreviousEarlyRepaymentsCountBeforeLoan,
            'VerificationType':VerificationType,
            'LanguageCode':LanguageCode,
            'Gender':Gender,
            'Country':Country,
            'UseOfLoan':UseOfLoan,
            'Education':Education,
            'MaritalStatus':MaritalStatus,
            'EmploymentStatus':EmploymentStatus,
            'EmploymentDurationCurrentEmployer':EmploymentDurationCurrentEmployer,
            'OccupationArea':OccupationArea,
            'HomeOwnershipType':HomeOwnershipType,
            'Rating':Rating,
            'CreditScoreEsMicroL':CreditScoreEsMicroL,
            'InterestAndPenaltyBalance':InterestAndPenaltyBalance,
            'PrincipalBalance':PrincipalBalance

            }
    features = pd.DataFrame(data, index=[0])
    return features


df = InputData()

cat_cols = df.select_dtypes(exclude=['int64', 'float64']).columns
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
enc = OrdinalEncoder()
X_cat_enc = enc.fit_transform(df[cat_cols])
df[cat_cols] = X_cat_enc

st.subheader("Input Parameters:")
st.write(df)
GB = GradientBoostingClassifier()
model = Pipeline([('Scaler', StandardScaler()), ('GB', GradientBoostingClassifier())])

def GBModel():
    cat_cols = Data.select_dtypes(exclude=['int64', 'float64']).columns
    num_cols = Data.select_dtypes(include=['int64', 'float64']).columns
    X = Data.drop(['Default', 'EMI', 'ELA', 'ROI'], axis=1)
    y = Data['Default']
    enc = OrdinalEncoder()
    X_cat_enc = enc.fit_transform(X[cat_cols])
    X[cat_cols] = X_cat_enc
    sc = StandardScaler()
    X = sc.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    GB.fit(X_train,y_train)


st.subheader("The result:")


def Prediction():
    GBModel()
    y_predict = GB.predict(df)
    return y_predict


if (st.sidebar.button("Predict")):
    val = Prediction()
    if (val == 0):
        st.write("Congratulations! The loan has been admitted :)")
    else:
        st.write("Unfortunately, The loan has been rejected :(")

    cat_cols = Data.select_dtypes(exclude=['int64', 'float64']).columns
    num_cols = Data.select_dtypes(include=['int64', 'float64']).columns
    X = Data.drop(['Default', 'EMI', 'ELA', 'ROI'], axis=1)
    y = Data[['EMI', 'ELA', 'ROI']]
    enc = OrdinalEncoder()
    X_cat_enc = enc.fit_transform(X[cat_cols])
    X[cat_cols] = X_cat_enc

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size = .75)

    transformer2 = ColumnTransformer([('scaling', StandardScaler(), [0, 1, 2, 4, 5, 6, 7, 11, 12, 14, 15, 16])],
                                     remainder='passthrough')
    transformer3 = Ridge(alpha=2.0)
    Pipeline1 = make_pipeline(transformer2, transformer3)
    Pipeline1.fit(X_train, y_train)
    st.subheader("The Credit Risk:")
    st.subheader("EMI:")
    st.write(Pipeline1.predict(df)[0][0])
    st.subheader("ELA:")
    st.write(Pipeline1.predict(df)[0][1])
    st.subheader("ROI:")
    st.write(Pipeline1.predict(df)[0][2])

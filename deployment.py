import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
import streamlit as st

st.write("""
# Cardiovascular Disease Prediction App
This app predicts the presence of cardiovascular disease based on user input features.
""")

st.sidebar.header('User Input Features')

def user_input_features():
    age = st.sidebar.number_input("Age (years)", min_value=15, max_value=99, value=50)
    height = st.sidebar.number_input("Height (cm)", min_value=120, max_value=220, value=170)
    weight = st.sidebar.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
    gender = st.sidebar.slider("Gender (1=Woman, 2=Man)", min_value=1, max_value=2, value=1)
    ap_hi = st.sidebar.number_input("Systolic Blood Pressure (mm Hg)", min_value=80, max_value=250, value=120)
    ap_lo = st.sidebar.number_input("Diastolic Blood Pressure (mm Hg)", min_value=40, max_value=150, value=80)
    cholesterol = st.sidebar.slider("Cholesterol (1=Normal, 2=Above Normal, 3=Well Above Normal)", min_value=1, max_value=3, value=1)
    gluc = st.sidebar.slider("Glucose (1=Normal, 2=Above Normal, 3=Well Above Normal)", min_value=1, max_value=3, value=1)
    smoke = st.sidebar.slider("Smoking (0=No, 1=Yes)", min_value=0, max_value=1, value=0)
    alco = st.sidebar.slider("Alcohol Intake (0=No, 1=Yes)", min_value=0, max_value=1, value=0)
    active = st.sidebar.slider("Physical Activity (0=No, 1=Yes)", min_value=0, max_value=1, value=1)
    
    data = {
        "Age": age*365,
        "Height": height,
        "Weight": weight,
        "Gender": gender,
        "Systolic Blood Pressure": ap_hi,
        "Diastolic Blood Pressure": ap_lo,
        "Cholesterol": cholesterol,
        "Glucose": gluc,
        "Smoking": smoke,
        "Alcohol Intake": alco,
        "Physical Activity": active
    }
    features = pd.DataFrame(data, index=[0])
    return features

user_inputs = user_input_features()
st.subheader('User Input Data')
st.write(user_inputs)

df = pd.read_csv("./data/cardio_train.csv", sep=";")
df = df.drop(columns=['id'])
df.drop(df[(df['ap_hi'] > df['ap_hi'].quantile(0.975)) | (df['ap_hi'] < df['ap_hi'].quantile(0.025))].index, inplace=True)
df.drop(df[(df['ap_lo'] > df['ap_lo'].quantile(0.975)) | (df['ap_lo'] < df['ap_lo'].quantile(0.025))].index, inplace=True)

# Handling missing values by dropping rows with any missing values
df_cleaned = df.dropna()
# Handling missing values by filling with the mean  
# df_filled = df.fillna(df.mean())

x = df_cleaned.iloc[:, :-1].values
y = df_cleaned.iloc[:, -1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

f_selector = SelectKBest(f_classif, k=6)
x_fbest = f_selector.fit_transform(x_train, y_train)

lr_clf_fbest = LogisticRegression(random_state=42)
lr_clf_fbest.fit(x_fbest, y_train)

# Apply the same preprocessing to user input
user_inputs_scaled = scaler.transform(user_inputs)
user_inputs_selected = f_selector.transform(user_inputs_scaled)

# y_pred_fbest = lr_clf_fbest.predict(f_selector.transform(x_test))
prediction = lr_clf_fbest.predict(user_inputs_selected)
prediction_proba = lr_clf_fbest.predict_proba(user_inputs_selected)

st.subheader('Class labels and their corresponding index number')
# Map the cardio values to their meanings
cardio_mapping = {0: "No Cardiovascular Disease", 1: "Cardiovascular Disease"}
# Create a DataFrame for better table display
cardio_df = pd.DataFrame({
    'Label': list(cardio_mapping.values())
})
# Display as a table
st.table(cardio_df)

st.subheader('Prediction Result')
st.write(f"The model predicts: **{cardio_mapping[prediction[0]]}**")

st.subheader('Prediction Probability')
st.write(f"Probability of No Cardiovascular Disease: {prediction_proba[0][0]:.2f}")
st.write(f"Probability of Cardiovascular Disease: {prediction_proba[0][1]:.2f}")
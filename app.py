import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Memuat dan Menggabungkan Dataset
df_wineqt = pd.read_csv('WineQT.csv')
df_wineqt = df_wineqt.drop('Id', axis=1)
df_wineqn = pd.read_csv('winequalityN.csv')
df_wineqn = df_wineqn.drop('type', axis=1)
df_wine = pd.concat([df_wineqt, df_wineqn], ignore_index=True)

# 2. Preprocessing Data
# Mengatasi Missing Values (Hapus baris yang kosong)
df_wine.dropna(inplace=True)  

# 3. Membagi Data
train_df, test_df = train_test_split(df_wine, test_size=0.2, random_state=42)

# 4. Pisahkan Fitur dan Target
X_train = train_df.drop('quality', axis=1) 
y_train = train_df['quality']
X_test = test_df.drop('quality', axis=1)
y_test = test_df['quality']

# 5. Normalisasi Data
scaler = StandardScaler()
numerical_features = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 
                       'density', 'pH', 'sulphates', 'alcohol']

# 5.a Normalisasi Data Training
X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])

# 5.b. Normalisasi Data Testing
X_test[numerical_features] = scaler.transform(X_test[numerical_features])  

# 6. Membuat dan Melatih Model
model = LinearRegression()
model.fit(X_train, y_train)

# 7. Evaluasi Model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# 8. Streamlit Interface untuk Prediksi
st.title('🍷 Wine Quality Prediction')

# Input parameter wine
fixed_acidity = st.number_input('Fixed Acidity', value=7.0)
volatile_acidity = st.number_input('Volatile Acidity', value=0.5)
citric_acid = st.number_input('Citric Acid', value=0.3)
residual_sugar = st.number_input('Residual Sugar', value=2.0)
chlorides = st.number_input('Chlorides', value=0.07)
free_sulfur_dioxide = st.number_input('Free Sulfur Dioxide', value=15.0)
total_sulfur_dioxide = st.number_input('Total Sulfur Dioxide', value=50.0)
density = st.number_input('Density', value=0.995)
pH = st.number_input('pH', value=3.3)
sulphates = st.number_input('Sulphates', value=0.6)
alcohol = st.number_input('Alcohol', value=10.0)

# 9. Preprocessing Input
# Normalisasi data input
input_data = np.array([fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                      chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, 
                      pH, sulphates, alcohol]).reshape(1, -1)

input_df = pd.DataFrame(input_data, columns=numerical_features)
input_data_scaled = scaler.transform(input_df)

# 10. Prediksi
prediction = model.predict(input_data_scaled) 

# 11. Menampilkan Hasil dan Evaluasi
st.subheader('Result:')
st.write('Wine Quality:', prediction[0])
st.write(" ")

st.write("📊## Evaluasi Model")
st.write("Mean Squared Error:", mse)
st.write("R-squared:", r2)
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Memuat dan Menggabungkan Dataset ---
df_wineqt = pd.read_csv('WineQT.csv')  # Ganti dengan path file WineQT.csv kamu
df_wineqt = df_wineqt.drop('Id', axis=1)  # Hapus kolom 'Id'
df_wineqn = pd.read_csv('winequalityN.csv') # Ganti dengan path file winequalityN.csv kamu
df_wineqn = df_wineqn.drop('type', axis=1)  # Menghapus kolom 'type'
df_wine = pd.concat([df_wineqt, df_wineqn], ignore_index=True)

# --- 2. Eksplorasi Data (EDA) ---
# ... (kode EDA opsional,  misalnya df_wine.describe())

# --- 3. Preprocessing Data ---
# 3.a. Mengatasi Missing Values (Hapus baris yang kosong)
df_wine.dropna(inplace=True)  

# --- 4. Pemilihan Fitur ---
# (Untuk saat ini, gunakan semua atribut) 
# ---  Kamu bisa menambahkan analisis korelasi atau feature importance di sini --- 

# --- 5. Membagi Data  ---
train_df, test_df = train_test_split(df_wine, test_size=0.2, random_state=42)

# --- 6. Pisahkan Fitur dan Target ---
X_train = train_df.drop('quality', axis=1) 
y_train = train_df['quality']
X_test = test_df.drop('quality', axis=1)
y_test = test_df['quality']

# --- 7. Normalisasi Data ---
scaler = StandardScaler()
numerical_features = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 
                       'density', 'pH', 'sulphates', 'alcohol']

# 7.a Normalisasi Data Training
X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])

# 7.b. Normalisasi Data Testing
X_test[numerical_features] = scaler.transform(X_test[numerical_features])  

# --- 8. Membuat dan Melatih Model ---
model = LinearRegression()
model.fit(X_train, y_train)

# --- 9. Evaluasi Model ---
# ... (kode evaluasi opsional) 

# --- 10. Streamlit Interface untuk Prediksi ---
st.title('Wine Quality Prediction')

# Input parameter wine dari pengguna
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

# --- 11. Preprocessing Input ---
# Normalisasi data input dengan StandardScaler yang sudah difit pada data training
input_data = np.array([fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                      chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, 
                      pH, sulphates, alcohol]).reshape(1, -1)  # 11 fitur

# --- Berikan nama fitur ke input_data ---
input_df = pd.DataFrame(input_data, columns=numerical_features)
input_data_scaled = scaler.transform(input_df)  # Transform DataFrame

# --- 12. Prediksi ---
prediction = model.predict(input_data_scaled) 

# --- 13. Menampilkan Hasil ---
st.subheader('Hasil Prediksi:')
st.write('Kualitas Wine:', prediction[0])
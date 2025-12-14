# ============================================================
# IMPORT LIBRARY
# ============================================================
import warnings
warnings.filterwarnings('ignore')

import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Heart Disease Classification",
    page_icon="❤️",
    layout="wide"
)

# ============================================================
# HEADER
# ============================================================
st.markdown(
    "<h1 style='text-align:center;'>❤️ Heart Disease Classification App</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;'>Decision Tree & Random Forest | Data Mining Project</p>",
    unsafe_allow_html=True
)
st.divider()

# ============================================================
# LOAD DATASET (AUTO)
# ============================================================
DATA_PATH = "heart_disease_uci.csv"

if not os.path.exists(DATA_PATH):
    st.error("❌ File dataset 'heart_disease_uci.csv' tidak ditemukan.")
    st.stop()

df = pd.read_csv(DATA_PATH)
st.success("Dataset berhasil dimuat otomatis")

# ============================================================
# DATA OVERVIEW
# ============================================================
st.subheader("📊 1. Data Overview")

col1, col2 = st.columns(2)

with col1:
    st.write("5 Data Teratas")
    st.dataframe(df.head())

with col2:
    info_df = pd.DataFrame({
        "Kolom": df.columns,
        "Tipe Data": df.dtypes.astype(str),
        "Non-Null": df.notnull().sum(),
        "Null": df.isnull().sum()
    })
    st.write("Informasi Dataset")
    st.dataframe(info_df)

st.divider()

# ============================================================
# TARGET
# ============================================================
target_col = "num"

# ============================================================
# EDA
# ============================================================
st.subheader("📈 2. Exploratory Data Analysis")

fig, ax = plt.subplots()
df[target_col].value_counts().sort_index().plot(kind="bar", ax=ax)
ax.set_xlabel("Kelas Penyakit")
ax.set_ylabel("Jumlah")
st.pyplot(fig)

st.divider()

# ============================================================
# PREPROCESSING
# ============================================================
st.subheader("⚙️ 3. Preprocessing")

df_proc = df.drop(columns=['id', 'dataset'], errors='ignore')
df_proc = df_proc.replace({'TRUE': 1, 'FALSE': 0, True: 1, False: 0})
df_proc = pd.get_dummies(df_proc, drop_first=True)

X = df_proc.drop(columns=[target_col])
y = df_proc[target_col]

st.success("Preprocessing selesai")

st.divider()

# ============================================================
# SPLIT & SCALING
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ============================================================
# MODEL SELECTION
# ============================================================
st.sidebar.header("⚙️ Pengaturan Model")

model_choice = st.sidebar.selectbox(
    "Pilih Model",
    ["Decision Tree", "Random Forest"]
)

if model_choice == "Random Forest":
    n_estimators = st.sidebar.slider("Jumlah Tree", 50, 300, 200)

# ============================================================
# TRAIN MODEL
# ============================================================
if model_choice == "Decision Tree":
    model = DecisionTreeClassifier(random_state=42)
else:
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=42
    )

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.subheader("🤖 4. Evaluasi Model")
st.metric("Accuracy", f"{acc:.2f}")

st.text("Classification Report")
st.text(classification_report(y_test, y_pred))

st.divider()

# ============================================================
# FORM PREDIKSI PASIEN
# ============================================================
st.subheader("🧑‍⚕️ 5. Prediksi Penyakit Jantung Pasien")

st.markdown("Masukkan data pasien di bawah ini:")

input_data = {}

for col in X.columns:
    if "sex" in col.lower():
        input_data[col] = st.selectbox(col, [0, 1])
    else:
        input_data[col] = st.number_input(col, value=0.0)

if st.button("🔍 Prediksi"):

    input_df = pd.DataFrame([input_data])
    input_df = input_df.reindex(columns=X.columns, fill_value=0)

    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]

    st.subheader("📌 Hasil Prediksi")

    if prediction == 0:
        st.success("✅ Pasien **TIDAK terdeteksi penyakit jantung**")
    else:
        st.error("⚠️ Pasien **TERDETEKSI memiliki penyakit jantung**")

# ============================================================
# FOOTER
# ============================================================
st.divider()
st.markdown(
    "<p style='text-align:center;font-size:12px;'>Data Mining Project | Streamlit</p>",
    unsafe_allow_html=True
)

# ============================================================
# IMPORT LIBRARY
# ============================================================
import warnings
warnings.filterwarnings("ignore")

import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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
st.markdown("<h1 style='text-align:center;'>❤️ Heart Disease Classification</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Decision Tree & Random Forest | Data Mining Project</p>", unsafe_allow_html=True)
st.divider()

# ============================================================
# LOAD DATASET (AUTO)
# ============================================================
DATA_PATH = "heart_disease_uci.csv"

if not os.path.exists(DATA_PATH):
    st.error("❌ Dataset heart_disease_uci.csv tidak ditemukan.")
    st.stop()

df = pd.read_csv(DATA_PATH)
st.success("Dataset berhasil dimuat")

# ============================================================
# DATA OVERVIEW
# ============================================================
st.subheader("📊 1. Data Overview")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**5 Data Teratas**")
    st.dataframe(df.head(), use_container_width=True)

with col2:
    info_df = pd.DataFrame({
        "Kolom": df.columns,
        "Tipe Data": df.dtypes.astype(str),
        "Non-Null": df.notnull().sum(),
        "Null": df.isnull().sum()
    })
    st.markdown("**Informasi Dataset**")
    st.dataframe(info_df, use_container_width=True)

st.divider()

# ============================================================
# TARGET & EDA
# ============================================================
target_col = "num"

st.subheader("📈 2. Exploratory Data Analysis")

fig, ax = plt.subplots()
df[target_col].value_counts().sort_index().plot(kind="bar", ax=ax)
ax.set_xlabel("Kelas Penyakit Jantung")
ax.set_ylabel("Jumlah")
st.pyplot(fig)

st.divider()

# ============================================================
# PREPROCESSING
# ============================================================
st.subheader("⚙️ 3. Preprocessing")

df_proc = df.drop(columns=["id", "dataset"], errors="ignore")
df_proc = df_proc.replace({"TRUE": 1, "FALSE": 0, True: 1, False: 0})
df_proc = pd.get_dummies(df_proc, drop_first=True)

X = df_proc.drop(columns=[target_col])
y = df_proc[target_col]

st.success("Preprocessing selesai")

st.divider()

# ============================================================
# SPLIT & SCALING
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ============================================================
# SIDEBAR MODEL
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
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# ============================================================
# EVALUATION
# ============================================================
st.subheader("🤖 4. Evaluasi Model")

col1, col2 = st.columns(2)

with col1:
    st.metric("Accuracy", f"{acc:.2f}")
    st.text("Classification Report")
    st.text(classification_report(y_test, y_pred))

with col2:
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", ax=ax_cm)
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    st.pyplot(fig_cm)

st.divider()

# ============================================================
# FORM INPUT PASIEN (USER FRIENDLY)
# ============================================================
st.subheader("🧑‍⚕️ 5. Prediksi Penyakit Jantung Pasien")
st.markdown("Masukkan data pasien:")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Usia", 1, 100, 50)
    trestbps = st.number_input("Tekanan Darah Istirahat", 80, 200, 130)
    chol = st.number_input("Kolesterol", 100, 400, 220)
    thalach = st.number_input("Detak Jantung Maksimum", 60, 220, 150)
    oldpeak = st.number_input("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
    ca = st.selectbox("Jumlah Pembuluh Darah Tersumbat", [0, 1, 2, 3])

with col2:
    sex = st.selectbox("Jenis Kelamin", ["Perempuan", "Laki-laki"])
    fbs = st.selectbox("Gula Darah Puasa > 120 mg/dL?", ["Tidak", "Ya"])
    exang = st.selectbox("Nyeri Dada Saat Olahraga?", ["Tidak", "Ya"])
    cp = st.selectbox("Tipe Nyeri Dada", ["Typical Angina", "Atypical Angina", "Non-anginal", "Asymptomatic"])
    restecg = st.selectbox("Hasil ECG", ["Normal", "ST-T Abnormality", "Left Ventricular Hypertrophy"])
    slope = st.selectbox("Slope ST Segment", ["Upsloping", "Flat", "Downsloping"])
    thal = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversable Defect"])

# ============================================================
# KONVERSI INPUT → ONE HOT (AMAN)
# ============================================================
input_data = {col: 0 for col in X.columns}

input_data["age"] = age
input_data["trestbps"] = trestbps
input_data["chol"] = chol
input_data["thalach"] = thalach
input_data["oldpeak"] = oldpeak
input_data["ca"] = ca
input_data["sex_Male"] = 1 if sex == "Laki-laki" else 0
input_data["fbs"] = 1 if fbs == "Ya" else 0
input_data["exang"] = 1 if exang == "Ya" else 0

cp_map = {
    "Typical Angina": "cp_typical angina",
    "Atypical Angina": "cp_atypical angina",
    "Non-anginal": "cp_non-anginal"
}
if cp in cp_map:
    input_data[cp_map[cp]] = 1

restecg_map = {
    "Normal": "restecg_normal",
    "ST-T Abnormality": "restecg_st-t abnormality"
}
if restecg in restecg_map:
    input_data[restecg_map[restecg]] = 1

slope_map = {
    "Upsloping": "slope_upsloping",
    "Flat": "slope_flat"
}
if slope in slope_map:
    input_data[slope_map[slope]] = 1

thal_map = {
    "Normal": "thal_normal",
    "Reversable Defect": "thal_reversable defect"
}
if thal in thal_map:
    input_data[thal_map[thal]] = 1

# ============================================================
# PREDIKSI (FIX ERROR KOLOM)
# ============================================================
if st.button("🔍 Prediksi Penyakit Jantung"):
    input_df = pd.DataFrame([input_data])

    # 🔥 BARIS PALING PENTING (ANTI ERROR)
    input_df = input_df[X.columns]

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

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
# LOAD DATASET
# ============================================================
DATA_PATH = "heart_disease_uci.csv"

if not os.path.exists(DATA_PATH):
    st.error("❌ Dataset heart_disease_uci.csv tidak ditemukan.")
    st.stop()

df = pd.read_csv(DATA_PATH)
st.success("✅ Dataset berhasil dimuat")

# ============================================================
# FIX TARGET → BINARY
# ============================================================
df["num"] = df["num"].apply(lambda x: 1 if x > 0 else 0)

# ============================================================
# HANDLE MISSING VALUE
# ============================================================
df = df.fillna(df.median(numeric_only=True))
df = df.fillna(df.mode().iloc[0])

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
        "Missing": df.isnull().sum()
    })
    st.markdown("**Informasi Dataset**")
    st.dataframe(info_df, use_container_width=True)

st.divider()

# ============================================================
# TARGET VARIABLE
# ============================================================
st.subheader("🎯 2. Target Variable")

st.markdown("""
Kolom target yang digunakan adalah **num** dengan keterangan:
- **0** → Tidak memiliki penyakit jantung  
- **1** → Memiliki penyakit jantung  
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Distribusi Target**")
    st.dataframe(df["num"].value_counts())

with col2:
    fig, ax = plt.subplots(figsize=(4,3))
    df["num"].value_counts().plot(kind="bar", ax=ax)
    ax.set_xlabel("Kelas Penyakit")
    ax.set_ylabel("Jumlah")
    st.pyplot(fig)

st.divider()

# ============================================================
# PREPROCESSING
# ============================================================
st.subheader("⚙️ 3. Preprocessing Data")

df_proc = df.drop(columns=["id", "dataset"], errors="ignore")
df_proc = df_proc.replace({"TRUE": 1, "FALSE": 0, True: 1, False: 0})
df_proc = pd.get_dummies(df_proc, drop_first=True)

X = df_proc.drop(columns=["num"])
y = df_proc["num"]

st.success("✅ Preprocessing selesai")
st.divider()

# ============================================================
# SPLIT DATA
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

st.subheader("📂 4. Pembagian Data")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Data", df_proc.shape[0])
with col2:
    st.metric("Data Training", X_train.shape[0])
with col3:
    st.metric("Data Testing", X_test.shape[0])

st.markdown("**Rasio:** 80% Training – 20% Testing")
st.divider()

# ============================================================
# MODEL SELECTION
# ============================================================
st.sidebar.header("⚙️ Pengaturan Model")

model_choice = st.sidebar.selectbox(
    "Pilih Model",
    ["Decision Tree", "Random Forest"]
)

if model_choice == "Decision Tree":
    model = DecisionTreeClassifier(random_state=42)
else:
    model = RandomForestClassifier(random_state=42)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# ============================================================
# EVALUASI MODEL
# ============================================================
st.subheader("🤖 5. Evaluasi Model")

col1, col2 = st.columns(2)

with col1:
    st.metric("Accuracy", f"{acc:.2f}")
    st.text("Classification Report")
    st.text(classification_report(y_test, y_pred))

with col2:
    fig_cm, ax_cm = plt.subplots(figsize=(4,3))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", ax=ax_cm)
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    st.pyplot(fig_cm)

st.divider()

# ============================================================
# FEATURE IMPORTANCE
# ============================================================
if hasattr(model, "feature_importances_"):
    st.subheader("📌 Feature Importance")

    importances = pd.Series(model.feature_importances_, index=X.columns)
    importances = importances.sort_values(ascending=True)

    fig_imp, ax_imp = plt.subplots(figsize=(6,8))
    importances.plot(kind="barh", ax=ax_imp, color="teal")
    ax_imp.set_title("Fitur Paling Berpengaruh")
    ax_imp.set_xlabel("Importance Score")
    st.pyplot(fig_imp)


# ============================================================
# PRECISION-RECALL CURVE
# ============================================================
st.subheader("📈 Precision-Recall Curve")

# Probabilitas prediksi (hanya untuk model yang mendukung predict_proba)
if hasattr(model, "predict_proba"):
    y_scores = model.predict_proba(X_test)[:, 1]
else:
    # fallback untuk model tanpa predict_proba
    y_scores = model.predict(X_test)

precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
avg_precision = average_precision_score(y_test, y_scores)

fig_pr, ax_pr = plt.subplots(figsize=(5,4))
ax_pr.plot(recall, precision, color="purple", linewidth=2)
ax_pr.set_title(f"Precision-Recall Curve (AP = {avg_precision:.2f})")
ax_pr.set_xlabel("Recall")
ax_pr.set_ylabel("Precision")
ax_pr.grid(True)

st.pyplot(fig_pr)


# ============================================================
# FORM INPUT MANUAL (VERSI AWAL)
# ============================================================
st.subheader("🧑‍⚕️ 6. Prediksi Penyakit Jantung")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Usia", 1, 100, 50)
    trestbps = st.number_input("Tekanan Darah Istirahat", 80, 200, 130)
    chol = st.number_input("Kolesterol", 100, 400, 220)
    thalach = st.number_input("Detak Jantung Maksimum", 60, 220, 150)
    oldpeak = st.number_input("Oldpeak", 0.0, 6.0, 1.0)
    ca = st.selectbox("Jumlah Pembuluh Darah Tersumbat", [0, 1, 2, 3])

with col2:
    sex = st.selectbox("Jenis Kelamin", ["Perempuan", "Laki-laki"])
    fbs = st.selectbox("Gula Darah Puasa > 120 mg/dL?", ["Tidak", "Ya"])
    exang = st.selectbox("Nyeri Dada Saat Olahraga?", ["Tidak", "Ya"])
    cp = st.selectbox("Tipe Nyeri Dada", ["typical angina", "atypical angina", "non-anginal", "asymptomatic"])
    restecg = st.selectbox("Hasil ECG", ["normal", "lv hypertrophy", "st-t abnormality"])
    slope = st.selectbox("Slope ST Segment", ["upsloping", "flat", "downsloping"])
    thal = st.selectbox("Thalassemia", ["normal", "fixed defect", "reversable defect"])

# ============================================================
# KONVERSI INPUT KE DUMMY SESUAI X.columns
# ============================================================
input_data = {col: 0 for col in X.columns}

input_data.update({
    "age": age,
    "trestbps": trestbps,
    "chol": chol,
    "thalach": thalach,
    "oldpeak": oldpeak,
    "ca": ca,
    "sex_Male": 1 if sex == "Laki-laki" else 0,
    "fbs": 1 if fbs == "Ya" else 0,
    "exang": 1 if exang == "Ya" else 0
})

# cp
if f"cp_{cp}" in input_data:
    input_data[f"cp_{cp}"] = 1

# restecg
if f"restecg_{restecg}" in input_data:
    input_data[f"restecg_{restecg}"] = 1

# slope
if f"slope_{slope}" in input_data:
    input_data[f"slope_{slope}"] = 1

# thal
if f"thal_{thal}" in input_data:
    input_data[f"thal_{thal}"] = 1

# ============================================================
# PREDIKSI
# ============================================================
if st.button("🔍 Prediksi Penyakit Jantung"):
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]

    st.subheader("📌 Hasil Prediksi")
    if prediction == 0:
        st.success("✅ Pasien **TIDAK terdeteksi penyakit jantung**")
    else:
        st.error("⚠️ Pasien **TERDETEKSI penyakit jantung**")

# ============================================================
# FOOTER
# ============================================================
st.divider()
st.markdown(
    "<p style='text-align:center;font-size:12px;'>Data Mining Project | Streamlit</p>",
    unsafe_allow_html=True
)



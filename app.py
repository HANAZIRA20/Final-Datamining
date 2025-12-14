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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_recall_curve, average_precision_score
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

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Distribusi Target**")
    st.dataframe(df["num"].value_counts())

with col2:
    fig, ax = plt.subplots(figsize=(3.5,2.5))
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

# Tampilkan fitur untuk memastikan
st.write("🔍 Kolom fitur yang digunakan untuk prediksi:")
st.write(list(X.columns))

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
    fig_cm, ax_cm = plt.subplots(figsize=(3.5,2.5))
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

    fig_imp, ax_imp = plt.subplots(figsize=(5,6))
    importances.plot(kind="barh", ax=ax_imp, color="teal")
    ax_imp.set_title("Fitur Paling Berpengaruh")
    ax_imp.set_xlabel("Importance Score")
    st.pyplot(fig_imp)

st.divider()

# ============================================================
# PRECISION-RECALL CURVE
# ============================================================
st.subheader("📈 Precision-Recall Curve")

if hasattr(model, "predict_proba"):
    y_scores = model.predict_proba(X_test)[:, 1]
else:
    y_scores = model.predict(X_test)

precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
avg_precision = average_precision_score(y_test, y_scores)

fig_pr, ax_pr = plt.subplots(figsize=(3.5,2.5))
ax_pr.plot(recall, precision, color="purple", linewidth=2)
ax_pr.set_title(f"Precision-Recall Curve (AP = {avg_precision:.2f})")
ax_pr.set_xlabel("Recall")
ax_pr.set_ylabel("Precision")
ax_pr.grid(True)

st.pyplot(fig_pr)

st.divider()

# ============================================================
# FORM INPUT MANUAL
# ============================================================
st.subheader("🧑‍⚕️ 6. Prediksi Penyakit Jantung")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Usia", 1, 100, 50)
    trestbps = st.number_input("Tekanan Darah Istirahat", 80, 200, 130)
    chol = st.number_input("Kolesterol", 100, 400, 220)
    # di dataset: 'thalch' → tapi di fitur: 'thalch' TIDAK ADA, yang ada 'thalch'? 
    # dari X.columns kamu: ada 'thalch'
    thalch = st.number_input("Detak Jantung Maksimum (thalch)", 60, 220, 150)
    oldpeak = st.number_input("Oldpeak", 0.0, 6.0, 1.0)
    ca = st.selectbox("Jumlah Pembuluh Darah Tersumbat", [0, 1, 2, 3])

with col2:
    sex = st.selectbox("Jenis Kelamin", ["Perempuan", "Laki-laki"])
    fbs = st.selectbox("Gula Darah Puasa > 120 mg/dL?", ["Tidak", "Ya"])
    exang = st.selectbox("Nyeri Dada Saat Olahraga?", ["Tidak", "Ya"])

    # ✅ Sesuaikan dengan dummy yang ADA:
    # cp_atypical angina, cp_non-anginal, cp_typical angina
    cp = st.selectbox("Tipe Nyeri Dada", [
        "typical angina",
        "atypical angina",
        "non-anginal"
    ])

    # ✅ Sesuaikan dengan dummy yang ADA:
    # restecg_normal, restecg_st-t abnormality
    restecg = st.selectbox("Hasil ECG", [
        "normal",
        "st-t abnormality"
    ])

    # slope_flat, slope_upsloping (tidak ada downsloping di fitur)
    slope = st.selectbox("Slope ST Segment", [
        "flat",
        "upsloping"
    ])

    # ✅ Sesuaikan dengan dummy yang ADA:
    # thal_normal, thal_reversable defect
    thal = st.selectbox("Thalassemia", [
        "normal",
        "reversable defect"
    ])

# ============================================================
# KONVERSI INPUT KE DUMMY (SINKRON DENGAN X.columns)
# ============================================================
input_data = {col: 0 for col in X.columns}

# numeric
input_data["age"] = age
input_data["trestbps"] = trestbps
input_data["chol"] = chol
input_data["fbs"] = 1 if fbs == "Ya" else 0
input_data["thalch"] = thalch
input_data["exang"] = 1 if exang == "Ya" else 0
input_data["oldpeak"] = oldpeak
input_data["ca"] = ca

# binary
input_data["sex_Male"] = 1 if sex == "Laki-laki" else 0

# categorical dummies – HARUS COCOK DENGAN NAMA FITUR

# cp
cp_col = f"cp_{cp}"
if cp_col in input_data:
    input_data[cp_col] = 1

# restecg
restecg_col = f"restecg_{restecg}"
if restecg_col in input_data:
    input_data[restecg_col] = 1

# slope
slope_col = f"slope_{slope}"
if slope_col in input_data:
    input_data[slope_col] = 1

# thal
thal_col = f"thal_{thal}"
if thal_col in input_data:
    input_data[thal_col] = 1

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

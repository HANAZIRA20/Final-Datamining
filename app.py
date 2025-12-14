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
# TARGET VARIABLE
# ============================================================
st.subheader("🎯 2. Distribusi Target (num)")

colA, colB = st.columns([1,1])

with colA:
    fig, ax = plt.subplots(figsize=(3,2))
    df["num"].value_counts().plot(kind="bar", ax=ax, color=["green","red"])
    ax.set_title("Distribusi Target")
    ax.set_xlabel("Kelas")
    ax.set_ylabel("Jumlah")
    st.pyplot(fig)

with colB:
    st.markdown("""
    ### 📘 Penjelasan Target (num)
    - **0** → Pasien **tidak** memiliki penyakit jantung  
    - **1** → Pasien **memiliki** penyakit jantung  
    - Grafik ini menunjukkan **jumlah masing-masing kelas**.
    - Jika kelas 1 lebih sedikit → dataset **imbalanced**, sehingga PR Curve lebih cocok daripada ROC.
    """)

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

col1, col2 = st.columns([1,1])

with col1:
    st.metric("Accuracy", f"{acc:.2f}")
    st.text("Classification Report")
    st.text(classification_report(y_test, y_pred))

with col2:
    fig_cm, ax_cm = plt.subplots(figsize=(3,2))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", ax=ax_cm)
    ax_cm.set_title("Confusion Matrix")
    st.pyplot(fig_cm)

st.markdown("""
### 📘 Penjelasan Confusion Matrix
- **True Positive (TP)** → Model benar memprediksi pasien sakit  
- **True Negative (TN)** → Model benar memprediksi pasien sehat  
- **False Positive (FP)** → Model salah memprediksi pasien sehat sebagai sakit  
- **False Negative (FN)** → Model salah memprediksi pasien sakit sebagai sehat  
- FN sangat penting di dunia medis karena pasien sakit bisa tidak terdeteksi.
""")

st.divider()

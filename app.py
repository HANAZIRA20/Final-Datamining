# ============================================================
# IMPORT LIBRARY
# ============================================================
import warnings
warnings.filterwarnings('ignore')

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
# STREAMLIT CONFIG
# ============================================================
st.set_page_config(
    page_title="Heart Disease Classification",
    layout="wide"
)

st.title("💓 Heart Disease Classification App")
st.write("Klasifikasi Penyakit Jantung menggunakan Decision Tree & Random Forest")

# ============================================================
# LOAD DATASET
# ============================================================
st.sidebar.header("📂 Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload file CSV", type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("Dataset berhasil diupload!")
else:
    st.info("Menggunakan dataset default")
    df = pd.read_csv("heart_disease_uci.csv")

# ============================================================
# PREVIEW DATA
# ============================================================
st.subheader("📊 Preview Dataset")
st.dataframe(df.head())

st.subheader("ℹ️ Info Dataset")
st.text(df.info())

st.subheader("❓ Missing Value per Kolom")
st.write(df.isnull().sum())

# ============================================================
# TARGET COLUMN
# ============================================================
if 'num' not in df.columns:
    st.error("Kolom target 'num' tidak ditemukan!")
    st.stop()

target_col = 'num'
st.success(f"Target terdeteksi: {target_col}")

# ============================================================
# EDA – DISTRIBUSI TARGET
# ============================================================
st.subheader("📈 Distribusi Target")

fig, ax = plt.subplots()
df[target_col].value_counts().sort_index().plot(kind='bar', ax=ax)
ax.set_xlabel("Kelas Penyakit Jantung")
ax.set_ylabel("Jumlah")
st.pyplot(fig)

# ============================================================
# HEATMAP KORELASI
# ============================================================
st.subheader("🔥 Correlation Heatmap (Numeric Only)")

numeric_df = df.select_dtypes(include=['int64', 'float64'])

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(numeric_df.corr(), cmap='coolwarm', ax=ax)
st.pyplot(fig)

# ============================================================
# PREPROCESSING
# ============================================================
st.subheader("⚙️ Preprocessing Data")

# Drop kolom yang tidak perlu
df = df.drop(columns=['id', 'dataset'], errors='ignore')

# Encode TRUE / FALSE ke 1 / 0
df = df.replace({
    'TRUE': 1, 'FALSE': 0,
    True: 1, False: 0
})

# One-hot encoding
df = pd.get_dummies(df, drop_first=True)

st.write("Jumlah kolom setelah encoding:", df.shape[1])

# ============================================================
# SPLIT DATA
# ============================================================
X = df.drop(columns=[target_col])
y = df[target_col]

test_size = st.sidebar.slider("Test Size", 0.1, 0.4, 0.2)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=test_size,
    random_state=42,
    stratify=y
)

# ============================================================
# NORMALISASI
# ============================================================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

st.success("Preprocessing selesai!")

# ============================================================
# MODEL SELECTION
# ============================================================
st.sidebar.header("🤖 Model")

model_choice = st.sidebar.selectbox(
    "Pilih Model",
    ["Decision Tree", "Random Forest"]
)

# ============================================================
# TRAIN MODEL
# ============================================================
if st.sidebar.button("🚀 Train Model"):

    if model_choice == "Decision Tree":
        model = DecisionTreeClassifier(random_state=42)

    else:
        n_estimators = st.sidebar.slider(
            "Jumlah Trees (Random Forest)",
            50, 300, 200
        )
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=42
        )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # ========================================================
    # EVALUATION
    # ========================================================
    st.subheader("📌 Hasil Evaluasi Model")

    acc = accuracy_score(y_test, y_pred)
    st.metric("Accuracy", f"{acc:.2f}")

    st.subheader("📄 Classification Report")
    st.text(classification_report(y_test, y_pred))

    st.subheader("📊 Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(
        confusion_matrix(y_test, y_pred),
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    st.success("Training & evaluasi selesai ✅")

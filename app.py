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
# SIDEBAR – DATASET
# ============================================================
st.sidebar.header("📂 Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload dataset (CSV)", type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("Dataset berhasil diupload")
else:
    if os.path.exists("heart_disease_uci.csv"):
        df = pd.read_csv("heart_disease_uci.csv")
        st.sidebar.info("Menggunakan dataset default")
    else:
        st.error("Dataset tidak ditemukan. Silakan upload file CSV.")
        st.stop()

# ============================================================
# SECTION 1 – DATA OVERVIEW
# ============================================================
st.subheader("📊 1. Data Overview")

col1, col2 = st.columns(2)

with col1:
    st.write("**5 Data Teratas**")
    st.dataframe(df.head())

with col2:
    st.write("**Informasi Dataset**")
    buffer = []
    df.info(buf=buffer)
    st.text("\n".join(buffer))

st.write("**Missing Value per Kolom**")
st.dataframe(df.isnull().sum())

st.divider()

# ============================================================
# TARGET COLUMN
# ============================================================
if 'num' not in df.columns:
    st.error("Kolom target 'num' tidak ditemukan!")
    st.stop()

target_col = 'num'
st.success("Target terdeteksi: kolom 'num'")

# ============================================================
# SECTION 2 – EDA
# ============================================================
st.subheader("📈 2. Exploratory Data Analysis (EDA)")

col1, col2 = st.columns(2)

with col1:
    st.write("**Distribusi Kelas Target**")
    fig, ax = plt.subplots()
    df[target_col].value_counts().sort_index().plot(kind='bar', ax=ax)
    ax.set_xlabel("Kelas Penyakit Jantung")
    ax.set_ylabel("Jumlah")
    st.pyplot(fig)

with col2:
    st.write("**Correlation Heatmap (Numerik)**")
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    fig, ax = plt.subplots(figsize=(6,4))
    sns.heatmap(numeric_df.corr(), cmap='coolwarm', ax=ax)
    st.pyplot(fig)

st.divider()

# ============================================================
# SECTION 3 – PREPROCESSING
# ============================================================
st.subheader("⚙️ 3. Preprocessing")

df_proc = df.copy()

df_proc = df_proc.drop(columns=['id', 'dataset'], errors='ignore')

df_proc = df_proc.replace({
    'TRUE': 1, 'FALSE': 0,
    True: 1, False: 0
})

df_proc = pd.get_dummies(df_proc, drop_first=True)

st.write("**Dataset setelah preprocessing**")
st.dataframe(df_proc.head())

st.info(f"Total fitur setelah encoding: {df_proc.shape[1]}")

st.divider()

# ============================================================
# SECTION 4 – SPLIT DATA
# ============================================================
st.subheader("🔀 4. Data Splitting & Scaling")

X = df_proc.drop(columns=[target_col])
y = df_proc[target_col]

test_size = st.sidebar.slider("Test Size", 0.1, 0.4, 0.2)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=test_size,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

st.success("Data berhasil di-split & dinormalisasi")

st.write("Ukuran Data:")
st.write({
    "X_train": X_train.shape,
    "X_test": X_test.shape,
    "y_train": y_train.shape,
    "y_test": y_test.shape
})

st.divider()

# ============================================================
# SECTION 5 – MODEL TRAINING
# ============================================================
st.subheader("🤖 5. Model Training & Evaluation")

model_choice = st.sidebar.selectbox(
    "Pilih Model",
    ["Decision Tree", "Random Forest"]
)

if model_choice == "Random Forest":
    n_estimators = st.sidebar.slider("Jumlah Trees", 50, 300, 200)

train_btn = st.sidebar.button("🚀 Train Model")

if train_btn:

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

    st.metric("🎯 Accuracy", f"{acc:.2f}")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Classification Report**")
        st.text(classification_report(y_test, y_pred))

    with col2:
        st.write("**Confusion Matrix**")
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

else:
    st.info("Klik **Train Model** untuk melihat hasil prediksi")

# ============================================================
# FOOTER
# ============================================================
st.divider()
st.markdown(
    "<p style='text-align:center;font-size:12px;'>Data Mining Project | Streamlit App</p>",
    unsafe_allow_html=True
)

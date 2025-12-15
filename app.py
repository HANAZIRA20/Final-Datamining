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
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Anime Type Classification",
    page_icon="🎌",
    layout="wide"
)

# ============================================================
# HEADER
# ============================================================
st.markdown("<h1 style='text-align:center;'>🎌 Anime Type Classification</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Decision Tree & Random Forest | Data Mining Project</p>", unsafe_allow_html=True)
st.divider()

# ============================================================
# LOAD DATASET
# ============================================================
DATA_PATH = "anime.csv"

if not os.path.exists(DATA_PATH):
    st.error("❌ Dataset anime.csv tidak ditemukan.")
    st.stop()

df = pd.read_csv(DATA_PATH)
st.success("✅ Dataset berhasil dimuat")

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
st.subheader("🎯 2. Target Variable (Type Anime)")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Distribusi Target**")
    st.dataframe(df["type"].value_counts())

with col2:
    fig, ax = plt.subplots(figsize=(4,3))
    df["type"].value_counts().plot(kind="bar", ax=ax)
    ax.set_xlabel("Type Anime")
    ax.set_ylabel("Jumlah")
    st.pyplot(fig)

st.divider()

# ============================================================
# PREPROCESSING
# ============================================================
st.subheader("⚙️ 3. Preprocessing Data")

df_proc = df.copy()

# Hapus kolom tidak relevan
df_proc = df_proc.drop(columns=["anime_id", "name"], errors="ignore")

# Handle missing value
df_proc["episodes"] = df_proc["episodes"].replace("?", None)
df_proc["episodes"] = pd.to_numeric(df_proc["episodes"])
df_proc = df_proc.fillna(df_proc.median(numeric_only=True))

# Encode genre (pakai panjang genre)
df_proc["genre_count"] = df_proc["genre"].apply(lambda x: len(str(x).split(",")))
df_proc = df_proc.drop(columns=["genre"])

# Encode target
le = LabelEncoder()
df_proc["type_encoded"] = le.fit_transform(df_proc["type"])

X = df_proc.drop(columns=["type", "type_encoded"])
y = df_proc["type_encoded"]

st.write("🔍 Fitur yang digunakan:")
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
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

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
    st.text(classification_report(y_test, y_pred, target_names=le.classes_))

with col2:
    fig_cm, ax_cm = plt.subplots(figsize=(4,3))
    cm = confusion_matrix(y_test, y_pred)

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=le.classes_,
        yticklabels=le.classes_,
        ax=ax_cm
    )

    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    ax_cm.set_title("Confusion Matrix")
    st.pyplot(fig_cm)

st.divider()

# ============================================================
# FEATURE IMPORTANCE
# ============================================================
if hasattr(model, "feature_importances_"):
    st.subheader("📌 Feature Importance")

    importances = pd.Series(model.feature_importances_, index=X.columns)
    importances = importances.sort_values(ascending=True)

    fig_imp, ax_imp = plt.subplots(figsize=(4,4))
    importances.plot(kind="barh", ax=ax_imp)
    ax_imp.set_title("Feature Importance")
    st.pyplot(fig_imp)

st.divider()

# ============================================================
# PREDIKSI MANUAL
# ============================================================
st.subheader("🎮 6. Prediksi Tipe Anime")

col1, col2 = st.columns(2)

with col1:
    episodes = st.number_input("Jumlah Episode", 1, 2000, 12)
    rating = st.slider("Rating", 0.0, 10.0, 7.5)
    members = st.number_input("Jumlah Member", 0, 5000000, 50000)

with col2:
    genre_count = st.slider("Jumlah Genre", 1, 10, 2)

if st.button("🔍 Prediksi Tipe Anime"):
    input_df = pd.DataFrame([{
        "episodes": episodes,
        "rating": rating,
        "members": members,
        "genre_count": genre_count
    }])

    prediction = model.predict(input_df)[0]
    anime_type = le.inverse_transform([prediction])[0]

    st.success(f"🎯 Prediksi Tipe Anime: **{anime_type}**")

# ============================================================
# FOOTER
# ============================================================
st.divider()
st.markdown(
    "<p style='text-align:center;font-size:12px;'>Anime Data Mining Project | Streamlit</p>",
    unsafe_allow_html=True
)

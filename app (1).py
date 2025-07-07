import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# --- Sidebar Navigation ---
st.sidebar.title("Main Page")
page = st.sidebar.radio("Navigation", ["🏠 Main Page", "📊 Classification", "📈 Clustering"])

# --- Main Page ---
if page == "🏠 Main Page":
    st.title("Ujian Akhir Semester")
    st.subheader("Streamlit Apps")
    st.markdown("Collection of my apps deployed in Streamlit")
    st.markdown("**Nama:** Mizan Ikbar")
    st.markdown("**NIM:** 22146003")

# --- Classification Page ---
elif page == "📊 Classification":
    st.title("Klasifikasi Diabetes Menggunakan KNN")
    df = pd.read_csv("diabetes.csv")
    st.write("### Data Sample")
    st.dataframe(df.head())

    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    st.write("### Metrik Klasifikasi")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

    st.write("### Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)

    st.write("### Prediksi Data Baru")
    inputs = []
    for col in X.columns:
        value = st.number_input(f"Masukkan nilai untuk {col}", value=float(df[col].mean()))
        inputs.append(value)

    if st.button("Prediksi"):
        input_scaled = scaler.transform([inputs])
        prediction = knn.predict(input_scaled)[0]
        st.success(f"Hasil Prediksi: {'Diabetes' if prediction == 1 else 'Tidak Diabetes'}")

# --- Clustering Page ---
elif page == "📈 Clustering":
    st.title("Clustering Pelanggan Berdasarkan Gender, Age, Income, dan Spend Score")
    data = pd.read_csv("lokasi_gerai_kopi_clean.csv")
    st.write("### Data Sample")
    st.dataframe(data.head())

    le = LabelEncoder()
    data["gender_encoded"] = le.fit_transform(data["gender"])

    features = ["gender_encoded", "age", "income", "spend_score"]
    X = data[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
    data["Cluster"] = kmeans.fit_predict(X_scaled)

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X_scaled)
    data["PCA1"] = pca_result[:, 0]
    data["PCA2"] = pca_result[:, 1]

    st.write("### Visualisasi Clustering (PCA 2D)")
    fig, ax = plt.subplots()
    sns.scatterplot(data=data, x="PCA1", y="PCA2", hue="Cluster", palette="tab10", ax=ax)
    for i in range(len(data)):
        ax.text(data["PCA1"][i], data["PCA2"][i], str(data["cust_id"][i]), fontsize=7)
    st.pyplot(fig)

    st.write("### Prediksi Cluster Pelanggan Baru")
    cust_id_input = st.text_input("Masukkan ID Pelanggan (cust_id)", value="CUST123")
    gender_input = st.selectbox("Pilih Gender", ["Female", "Male"])
    gender_encoded = 0 if gender_input == "Female" else 1
    age_input = st.number_input("Usia (age)", value=float(data["age"].mean()))
    income_input = st.number_input("Pendapatan (income)", value=float(data["income"].mean()))
    spend_input = st.number_input("Skor Belanja (spend_score)", value=float(data["spend_score"].mean()))

    if st.button("Prediksi Cluster"):
        new_scaled = scaler.transform([[gender_encoded, age_input, income_input, spend_input]])
        new_cluster = kmeans.predict(new_scaled)[0]
        st.success(f"Pelanggan dengan ID: {cust_id_input} masuk ke dalam Cluster {new_cluster}")

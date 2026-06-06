import streamlit as st
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import plotly.express as px

from PIL import Image
from tensorflow.keras.utils import img_to_array

# =====================================================
# PAGE CONFIG
# =====================================================

import streamlit as st

if "login" not in st.session_state:
    st.session_state.login = False

st.set_page_config(page_title="NutriVision AI Pro", page_icon="🍜", layout="wide")

# =====================================================
# CUSTOM CSS
# =====================================================

def load_css():
    with open("style.css") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )

load_css()

# =====================================================
# LOAD MODEL
# =====================================================


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "models/model_makanan_indonesia.h5")

model = load_model()

# =====================================================
# NAMA KELAS
# =====================================================

class_names = [
    "Ayam Goreng",
    "Burger",
    "French Fries",
    "Gado-Gado",
    "Ikan Goreng",
    "Mie Goreng",
    "Nasi Goreng",
    "Nasi Padang",
    "Pizza",
    "Rawon",
    "Rendang",
    "Sate",
    "Soto",
]

# =====================================================
# DATABASE GIZI
# =====================================================

nutrisi = {
    "Ayam Goreng": {"kalori": 260, "protein": 25, "lemak": 15, "karbo": 8},
    "Burger": {"kalori": 295, "protein": 17, "lemak": 14, "karbo": 30},
    "French Fries": {"kalori": 312, "protein": 4, "lemak": 15, "karbo": 41},
    "Gado-Gado": {"kalori": 320, "protein": 12, "lemak": 18, "karbo": 26},
    "Ikan Goreng": {"kalori": 240, "protein": 23, "lemak": 14, "karbo": 0},
    "Mie Goreng": {"kalori": 380, "protein": 11, "lemak": 17, "karbo": 45},
    "Nasi Goreng": {"kalori": 333, "protein": 8, "lemak": 12, "karbo": 45},
    "Nasi Padang": {"kalori": 664, "protein": 28, "lemak": 30, "karbo": 68},
    "Pizza": {"kalori": 285, "protein": 12, "lemak": 10, "karbo": 36},
    "Rawon": {"kalori": 300, "protein": 20, "lemak": 12, "karbo": 18},
    "Rendang": {"kalori": 468, "protein": 26, "lemak": 36, "karbo": 9},
    "Sate": {"kalori": 320, "protein": 22, "lemak": 18, "karbo": 10},
    "Soto": {"kalori": 280, "protein": 18, "lemak": 8, "karbo": 22},
}

# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.image(
        "assets/logo.png",
        width=120
    )

    st.title("🥗 NutriVision AI")

    st.caption(
        "Food Recognition & Nutrition Analysis"
    )

    st.success("🟢 AI Online")

    st.markdown("---")

    st.metric(
        "Dataset",
        "13 Kelas"
    )

    st.metric(
        "Model",
        "CNN"
    )

    st.metric(
        "Akurasi",
        "92%"
    )

    st.markdown("---")

    st.info(
        "📸 Upload foto makanan untuk analisis nutrisi otomatis"
    )

    st.markdown("---")

    st.caption(
        "Version 1.0 | © 2026"
    )
    
    menu = st.sidebar.radio(
    "Menu",
    [
        "🏠 Home",
        "📖 Tentang",
        "📞 Kontak"
    ]
)
    
if menu == "🏠 Home":
    pass

elif menu == "📖 Tentang":
    st.title("Tentang Aplikasi")
    st.write("NutriVision AI adalah aplikasi deteksi makanan berbasis CNN.")

elif menu == "📞 Kontak":
    st.title("Kontak")
    st.write("Muhammad Aprizal")

# =====================================================
# HEADER
# =====================================================

st.markdown(
    """
<div class="hero">

<h1>🥗 NutriVision AI</h1>

<p>
Smart Food Recognition & Nutrition Scanner
</p>

</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="card">

<h3>📸 Cara Penggunaan</h3>

1. Ambil foto makanan
2. Upload atau Scan
3. AI akan menganalisis
4. Lihat informasi nutrisi

</div>
""",
    unsafe_allow_html=True,
)

# =====================================================
# UPLOAD
# =====================================================

st.markdown("""
### 📸 AI Food Scanner

Arahkan kamera ke makanan dan ambil foto untuk mendapatkan analisis nutrisi otomatis.
""")

tab1, tab2 = st.tabs(["📷 Scan Kamera", "🖼️ Upload Gambar"])

uploaded_file = None

with tab1:

    st.markdown("### 📷 Food Scanner")

    col_cam1, col_cam2, col_cam3 = st.columns([1,2,1])

with col_cam2:
        uploaded_file = st.camera_input("📷 Ambil Foto Makanan")

with tab2:
    upload = st.file_uploader("Upload gambar makanan", type=["jpg", "jpeg", "png"])

    if upload:
        uploaded_file = upload

st.markdown("""
### 📷 Food Scanner

Posisikan makanan di tengah kamera
untuk hasil analisis terbaik.
""")

# =====================================================
# DETEKSI
# =====================================================

if uploaded_file:

    img = Image.open(uploaded_file)

    col1, col2 = st.columns([0.8, 1.2])

    with col1:

        st.image(img, width=400)

        c1, c2, c3 = st.columns(3)

    with c1:
        if st.button("📷 Foto Baru"):
            st.rerun()

    with c2:
        if st.button("🔄 Scan Ulang"):
          st.rerun()

    with c3:
     st.button("💾 Simpan Hasil")

    scan = st.progress(0)

    for i in range(100):
        time.sleep(0.01)
        scan.progress(i + 1)
    
    
    colA, colB = st.columns(2)

    with colA:
        if st.button("🔄 Scan Lagi"):
            st.rerun()

    with colB:
        if st.button("🗑️ Reset"):
             st.rerun()

    img_resize = img.resize((224, 224))

    img_array = img_to_array(img_resize)

    img_array = np.expand_dims(img_array, axis=0)

    img_array = img_array / 255.0

    pred = model.predict(img_array, verbose=0)

    kelas = np.argmax(pred)

    confidence = float(np.max(pred) * 100)

    with col2:
        st.markdown(
            """
        <div class="result-card">
        <h3>🤖 Hasil Analisis AI</h3>
        </div>
        """,
            unsafe_allow_html=True,
        )

        nama_makanan = class_names[kelas]

        st.markdown(
            f"""
<div class="card">

<h2>
🍽️ {nama_makanan}
</h2>

<h3>
Confidence {confidence:.2f}%
</h3>

</div>
""",
            unsafe_allow_html=True,
        )

        st.progress(int(confidence))

        st.metric("🎯 Akurasi AI",f"{confidence:.2f}%")

        if confidence < 60:

            st.warning("⚠️ Makanan tidak terdaftar")

        else:

            st.success("✅ Makanan dikenali")

    st.subheader("🏆 Top 3 Prediksi")

    top3 = np.argsort(pred[0])[-3:][::-1]

    for idx in top3:

        persen = pred[0][idx] * 100

        st.progress(int(persen))

        st.write(f"🍴 {class_names[idx]} ({persen:.2f}%)")

    st.write(f"{class_names[idx]} " f"({pred[0][idx]*100:.2f}%)")

    if nama_makanan in nutrisi:

        data = nutrisi[nama_makanan]

        health_score = 100

        if data["kalori"] > 500:
            health_score -= 20

        if data["lemak"] > 20:
            health_score -= 15

    st.markdown(
        f"""
<div class="card">

<h3>💚 Health Score</h3>

<h1 style="color:#22c55e;">
{health_score}/100
</h1>

</div>
""",
        unsafe_allow_html=True,
    )

    if nama_makanan in nutrisi:

        data = nutrisi[nama_makanan]

    health_score = 100

    if data["kalori"] > 500:
        health_score -= 20

    if data["lemak"] > 20:
        health_score -= 15

        st.metric("💚 Health Score", f"{health_score}/100")

        if health_score >= 90:
            st.success("🥗 Sangat Baik Untuk Diet")

        elif health_score >= 70:
            st.info("🍽️ Masih Aman Dikonsumsi")

        else:
            st.warning("⚠️ Konsumsi Secukupnya")

    st.subheader("💡 Saran AI")

    if data["kalori"] > 500:

        st.warning("Kalori cukup tinggi. Cocok untuk bulking.")

    else:

        st.success("Kalori relatif ringan. Cocok untuk diet.")

    st.subheader("🍎 Informasi Gizi")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("🔥 Kalori", f"{data['kalori']} kcal")

    c2.metric("🥩 Protein", f"{data['protein']} g")

    c3.metric("🥑 Lemak", f"{data['lemak']} g")

    c4.metric("🍞 Karbo", f"{data['karbo']} g")

    chart_df = pd.DataFrame(
        {
            "Nutrisi": ["Protein", "Lemak", "Karbohidrat"],
            "Gram": [data["protein"], data["lemak"], data["karbo"]],
        }
    )

    st.subheader("📊 Grafik Nutrisi")

    fig = px.pie(
        chart_df,
        values="Gram",
        names="Nutrisi",
        hole=0.6
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("---")

# =====================
# TOMBOL RESET
# =====================

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    if st.button("🗑 Reset"):
        st.rerun()



# =====================
# FOOTER
# =====================

st.markdown("""
<hr>

<center>

<h4>
🥗 NutriVision AI Pro
</h4>

<p>
Powered by TensorFlow + Streamlit
</p>

</center>
""", unsafe_allow_html=True)

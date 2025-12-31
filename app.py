import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

model = joblib.load("model/svm_linear.pkl")
tfidf = joblib.load("model/tfidf.pkl")

# ======================
# CONFIG HALAMAN
# ======================
st.set_page_config(
    page_title="Analisis Sentimen PPKM",
    page_icon="💬",
    layout="centered"
)

# ======================
# LOAD MODEL
# ======================
model = joblib.load("model/svm_linear.pkl")
tfidf = joblib.load("model/tfidf.pkl")

stop_id = set(stopwords.words("indonesian"))

# ======================
# PREPROCESS
# ======================
def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_id]
    return " ".join(tokens)

# ======================
# HEADER
# ======================
st.markdown(
    "<h1 style='text-align:center;'>Analisis Sentimen Twitter PPKM</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center; color:gray;'>"
    "Aplikasi klasifikasi sentimen menggunakan <b>Support Vector Machine (SVM)</b>"
    "</p>",
    unsafe_allow_html=True
)

st.write("---")

# ======================
# INPUT
# ======================
st.markdown("### Masukkan Teks Tweet")
text_input = st.text_area(
    "",
    placeholder="Contoh: Kebijakan PPKM sangat membantu menekan penyebaran COVID-19",
    height=120
)

# ======================
# PREDIKSI
# ======================
if st.button("🔍 Prediksi Sentimen", use_container_width=True):
    if text_input.strip() == "":
        st.warning("⚠️ Teks tidak boleh kosong")
    else:
        clean_text = preprocess(text_input)
        text_tfidf = tfidf.transform([clean_text])
        prediction = model.predict(text_tfidf)[0]

        st.write("---")

        if prediction == 1:
            st.success("✅ Sentimen POSITIF")
        else:
            st.error("❌ Sentimen NEGATIF")

# ======================
# FOOTER
# ======================
st.write("---")
st.markdown(
    "<p style='text-align:center; font-size:12px; color:gray;'>"
    "TA Machine Learning – Analisis Sentimen Twitter PPKM"
    "</p>",
    unsafe_allow_html=True
)

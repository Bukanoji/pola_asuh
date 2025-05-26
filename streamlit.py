import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib                                    # ← baru
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE, RandomOverSampler
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# ─────────────────────────────────────────────────────────────
# Konfigurasi Global
st.set_page_config(layout="wide", page_title="Evaluasi Kernel RBF SVM Multi Rasio + Prediksi")
RANDOM_STATE = 42
LABEL_MAPPING = {'negatif': 0, 'netral': 1, 'positif': 2}
INV_LABEL_MAPPING = {v: k for k, v in LABEL_MAPPING.items()}
TARGET_NAMES = [INV_LABEL_MAPPING[i] for i in sorted(LABEL_MAPPING.values())]

# ─────────────────────────────────────────────────────────────
# Preprocessing
factory = StemmerFactory()
stemmer = factory.create_stemmer()
def preprocess_ind(text):
    if not isinstance(text, str): return ""
    txt = text.lower()
    txt = pd.Series(txt).str.replace(r"http\S+", " ", regex=True).iloc[0]
    txt = pd.Series(txt).str.replace(r"@\w+", " ", regex=True).iloc[0]
    txt = pd.Series(txt).str.replace(r"#\w+", " ", regex=True).iloc[0]
    txt = pd.Series(txt).str.replace(r"[^a-z\s]", " ", regex=True).iloc[0]
    return " ".join(stemmer.stem(w) for w in txt.split() if w.strip())

# ─────────────────────────────────────────────────────────────
# Load Data
@st.cache_data
def load_data(source_type, local_file_path=None, uploaded_file_obj=None):
    try:
        if source_type == "Lokal":
            return pd.read_csv(local_file_path)
        elif source_type == "Unggah" and uploaded_file_obj:
            return pd.read_csv(uploaded_file_obj)
    except Exception as e:
        st.error(f"Gagal memuat data: {e}")
    return None

# ─────────────────────────────────────────────────────────────
# Resampling helper
def get_sampler(y_series):
    min_c = y_series.value_counts().min()
    if min_c > 2:
        return SMOTE(random_state=RANDOM_STATE, k_neighbors=min(5, min_c-1))
    return RandomOverSampler(random_state=RANDOM_STATE)

# ─────────────────────────────────────────────────────────────
# Evaluasi RBF (TIDAK DIUBAH)
def evaluate_rbf_kernel(X_train, X_test, y_train, y_test, ratio_display):
    st.write(f"### Split Rasio: {ratio_display} | Kernel: RBF")
    vec = TfidfVectorizer()
    X_tr = vec.fit_transform(X_train)
    X_te = vec.transform(X_test)
    sampler = get_sampler(pd.Series(y_train))
    X_res, y_res = sampler.fit_resample(X_tr, y_train)
    svm = SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced', random_state=RANDOM_STATE)
    svm.fit(X_res, y_res)
    y_pred = svm.predict(X_te)
    acc = accuracy_score(y_test, y_pred)
    st.write(f"**Akurasi:** `{acc:.4f}`")
    rpt = classification_report(y_test, y_pred, target_names=TARGET_NAMES, output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(rpt).transpose().style.format("{:.2f}"))
    cm = confusion_matrix(y_test, y_pred, labels=list(LABEL_MAPPING.values()))
    fig, ax = plt.subplots(figsize=(4,4))
    ConfusionMatrixDisplay(cm, display_labels=TARGET_NAMES).plot(ax=ax)
    st.pyplot(fig)
    return acc

# ─────────────────────────────────────────────────────────────
# Load pre-trained pipeline untuk prediksi
@st.cache_resource
def load_trained_pipeline(path="sentiment_model.joblib"):
    return joblib.load(path)

# ─────────────────────────────────────────────────────────────
# Sidebar & Load Data
st.sidebar.header("⚙️ Pengaturan Data")
src = st.sidebar.radio("Pilih Sumber Data:", ("Lokal", "Unggah"))
file_obj = None; path = None
if src == "Lokal":
    path = "polasuh1output.csv"
else:
    file_obj = st.sidebar.file_uploader("Unggah CSV", type="csv")
df = load_data(src, path, file_obj)
if df is None or 'steming_data' not in df or 'sentimen_lexicon' not in df:
    st.error("Pastikan CSV ada dan memiliki kolom 'steming_data' & 'sentimen_lexicon'."); st.stop()

# ─────────────────────────────────────────────────────────────
# Judul & Evaluasi Multi-Split
st.title("📊 Evaluasi Kernel RBF SVM Multi Rasio + Prediksi Sentimen")
X_raw = df['steming_data'].fillna('').astype(str)
y_raw = df['sentimen_lexicon'].map(LABEL_MAPPING)
st.header("🔍 Evaluasi RBF Kernel pada Berbagai Rasio Split")
results = {}
for name, ts in {'90:10':0.1,'80:20':0.2,'70:30':0.3}.items():
    Xt, Xv, yt, yv = train_test_split(X_raw, y_raw, test_size=ts,
                                     stratify=y_raw, random_state=RANDOM_STATE)
    results[name] = evaluate_rbf_kernel(Xt, Xv, yt, yv, name)
st.subheader("📈 Ringkasan Akurasi (RBF Kernel)")
st.dataframe(pd.DataFrame.from_dict(results, orient='index', columns=['Akurasi']).style.format("{:.4f}"))

# ─────────────────────────────────────────────────────────────
# Prediksi Kalimat Baru
st.header("🔮 Prediksi Sentimen Kalimat Baru (Pre-trained)")
pipeline = load_trained_pipeline("sentiment_model.joblib")
teks_input = st.text_area("Masukkan Teks:")
if st.button("Prediksi"):
    if not teks_input.strip():
        st.warning("Silakan masukkan teks terlebih dahulu.")
    else:
        teks_proc = preprocess_ind(teks_input)
        pred_num = pipeline.predict([teks_proc])[0]
        st.success(f"Prediksi Sentimen: **{INV_LABEL_MAPPING[pred_num]}**")

st.markdown("---")
st.write("© 2025 Aplikasi — Semua hak cipta dilindungi.")

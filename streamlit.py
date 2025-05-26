import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC, SVC
from sklearn.kernel_approximation import Nystroem
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from joblib import Memory
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
import os
import traceback

# --- Tambahan Import untuk Sentiment Analysis ---
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.utils import class_weight

# --- Streamlit Page Configuration ---
st.set_page_config(layout="wide", page_title="Analisis Kernel SVM Multi-Rasio + Sentimen")

# --- Default Local File Paths ---
DEFAULT_LOCAL_CSV_FILE_PATH = r"polasuh1output.csv"
SENTIMENT_CSV_PATH = "polasuh1output.csv"  # Jika data sentimen terletak di file yang sama

# =============================================================================
# 0. Fungsi Praproses di Tingkat Modul (agar tidak menjadi fungsi lokal)
# =============================================================================
def preprocess_ind(text):
    """
    Pra-pemrosesan sederhana: 
    - ubah ke lowercase
    - hilangkan URL, mention, hashtag, angka, dan karakter non-alfabet
    - lakukan stemming Bahasa Indonesia
    """
    if not isinstance(text, str):
        return ""
    # Mengubah ke huruf kecil
    txt = text.lower()
    # Hapus URL
    txt = pd.Series(txt).str.replace(r"http\S+", " ", regex=True).iloc[0]
    # Hapus mention (@username)
    txt = pd.Series(txt).str.replace(r"@\w+", " ", regex=True).iloc[0]
    # Hapus hashtag (#tag)
    txt = pd.Series(txt).str.replace(r"#\w+", " ", regex=True).iloc[0]
    # Hapus karakter non-alfabet
    txt = pd.Series(txt).str.replace(r"[^a-z\s]", " ", regex=True).iloc[0]
    # Stemming tiap kata
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    return " ".join([stemmer.stem(w) for w in txt.split() if w.strip() != ""])

# =============================================================================
# 1. Fungsi Muat Data (load_data)
# =============================================================================
@st.cache_data(show_spinner=False)
def load_data(source_type: str, local_file_path: str = None, uploaded_file_obj=None):
    """
    Memuat data dari file lokal atau unggahan CSV. 
    Jika gagal, mengembalikan data dummy.
    """
    # Jika sumber lokal dipilih
    if source_type == "Lokal":
        try:
            df = pd.read_csv(local_file_path)
            return df, False
        except Exception as e:
            st.error(f"Gagal memuat file lokal '{local_file_path}': {e}")
            st.info("Beralih ke data dummy.")
            return create_dummy_data(), True

    # Jika sumber unggahan dipilih
    elif source_type == "Unggah":
        if uploaded_file_obj is not None:
            try:
                df = pd.read_csv(uploaded_file_obj)
                return df, False
            except Exception as e:
                st.error(f"Gagal memuat file unggahan: {e}")
                st.info("Beralih ke data dummy.")
                return create_dummy_data(), True
        else:
            st.warning("Tidak ada file CSV yang diunggah. Menggunakan data dummy.")
            return create_dummy_data(), True

    else:
        st.error(f"Tipe sumber data tidak dikenali: {source_type}. Menggunakan data dummy.")
        return create_dummy_data(), True

# =============================================================================
# 2. Fungsi untuk Muat dan Latih Model Sentimen (Linear SVC)
# =============================================================================
@st.cache_data(show_spinner=False)
def load_and_train_sentiment_model(csv_path: str):
    """
    1. Memuat data CSV berlabel (kolom 'full_text', 'sentimen_lexicon').
    2. Melakukan preprocessing dengan preprocess_ind (tingkat modul).
    3. Melatih ImbPipeline: TfidfVectorizer + LinearSVC (class_weight='balanced').
    4. Mengembalikan pipeline terlatih.
    """
    # --- 1. Muat Data ---
    try:
        df_sent = pd.read_csv(csv_path)
    except Exception as e:
        st.error(f"Gagal memuat file sentimen berlabel di '{csv_path}': {e}")
        return None

    # Pastikan kolom ada
    if 'full_text' not in df_sent.columns or 'sentimen_lexicon' not in df_sent.columns:
        st.error(f"File '{csv_path}' harus memiliki kolom 'full_text' dan 'sentimen_lexicon'.")
        return None

    # --- 2. Terapkan Preprocessing di Tingkat Modul ---
    df_sent["text_preprocessed"] = df_sent["full_text"].apply(preprocess_ind)

    # Label dan class_weight
    labels = df_sent["sentimen_lexicon"].values
    cw = class_weight.compute_class_weight(
        class_weight="balanced",
        classes=np.unique(labels),
        y=labels
    )
    class_weights_dict = {cls: cw_val for cls, cw_val in zip(np.unique(labels), cw)}

    # --- 3. Definisi Pipeline: TF-IDF + LinearSVC ---
    pipeline = ImbPipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_df=0.85,
            min_df=5,
            preprocessor=preprocess_ind
        )),
        ("clf", LinearSVC(
            class_weight=class_weights_dict,
            random_state=42,
            max_iter=5000
        ))
    ])

    # --- 4. Latih Model ---
    pipeline.fit(df_sent["text_preprocessed"], df_sent["sentimen_lexicon"])
    return pipeline

# Muat dan latih model sentimen sekali di awal (cache-enabled)
sentiment_pipeline = load_and_train_sentiment_model(SENTIMENT_CSV_PATH)

# =============================================================================
# 3. Fungsi-Fungsi Bantu (Tanpa Perubahan)
# =============================================================================
def create_dummy_data():
    """Membuat DataFrame dummy untuk demonstrasi."""
    data = {
        'steming_data': [
            'good text positive', 'bad text negative', 'neutral info',
            'another good one', 'very bad experience', 'just okay content',
            'excellent work', 'terrible service', 'average performance',
            'positive feedback example', 'negative comment here', 'neutral statement now',
            'happy with product', 'unhappy with service', 'no strong opinion',
            'more positive content', 'more negative content', 'more neutral content',
            'absolutely fantastic', 'truly awful', 'perfectly mediocre',
            'another positive example', 'yet another negative one', 'quite neutral indeed',
            'very good stuff', 'very bad stuff', 'very neutral stuff'
        ],
        'sentimen_lexicon': [
            'positif', 'negatif', 'netral',
            'positif', 'negatif', 'netral',
            'positif', 'negatif', 'netral',
            'positif', 'negatif', 'netral',
            'positif', 'negatif', 'netral',
            'positif', 'negatif', 'netral',
            'positif', 'negatif', 'netral',
            'positif', 'negatif', 'netral',
            'positif', 'negatif', 'netral'
        ]
    }
    df = pd.DataFrame(data)
    st.warning("Menggunakan data dummy. Hasil akan berdasarkan data ini.")
    return df

def get_sampler(y_train_series):
    min_count = y_train_series.value_counts().min()
    if min_count > 2:
        k_neighbors = min(5, min_count - 1)
        sampler = SMOTE(random_state=42, k_neighbors=k_neighbors)
        method = f"SMOTE (k_neighbors={k_neighbors})"
    else:
        sampler = RandomOverSampler(random_state=42)
        method = "RandomOverSampler"
    return sampler, method

def display_resampling_distributions_st(y_train_series, y_resampled_series, method_name, current_inv_label_mapping):
    st.write("**Distribusi Kelas Sebelum Resample:**")
    st.dataframe(y_train_series.map(current_inv_label_mapping).value_counts())
    st.write(f"**Metode Resampling Digunakan: {method_name}**")
    st.write("**Distribusi Kelas Setelah Resample:**")
    st.dataframe(pd.Series(y_resampled_series).map(current_inv_label_mapping).value_counts())

def plot_confusion_matrix_st(y_test, y_pred, labels_num, current_target_names):
    cm = confusion_matrix(y_test, y_pred, labels=labels_num)
    fig, ax = plt.subplots(figsize=(6, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=current_target_names)
    disp.plot(ax=ax, cmap=plt.cm.Blues, xticks_rotation='vertical')
    ax.set_title("Confusion Matrix")
    return fig

# --- Global Label Mapping ---
LABEL_MAPPING_GLOBAL = {'negatif': 0, 'netral': 1, 'positif': 2}
INV_LABEL_MAPPING_GLOBAL = {v: k for k, v in LABEL_MAPPING_GLOBAL.items()}
TARGET_NAMES_GLOBAL = [INV_LABEL_MAPPING_GLOBAL[i] for i in sorted(LABEL_MAPPING_GLOBAL.values())]

# =============================================================================
# 4. Fungsi Evaluasi Kernel (Tanpa Perubahan)
# =============================================================================
def evaluate_linear_kernel_st(X_train, X_test, y_train, y_test, ratio_name_for_display):
    st.write(f"#### Kernel: Linear | Split: {ratio_name_for_display}")
    y_train_series = pd.Series(y_train)
    sampler, method_name = get_sampler(y_train_series)

    pipeline = ImbPipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1,2), 
            stop_words='english', 
            max_df=0.95, 
            min_df=2, 
            max_features=5000
        )),
        ('sampler', sampler),
        ('clf', LinearSVC(
            class_weight='balanced', 
            max_iter=5000, 
            random_state=42, 
            dual="auto"
        ))
    ])
    param_grid = {'clf__C': [0.1, 1, 10, 100]}

    grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"**Akurasi:** `{accuracy:.4f}`")

    report_dict = classification_report(
        y_test, y_pred,
        target_names=TARGET_NAMES_GLOBAL,
        output_dict=True,
        zero_division=0
    )
    report_df = pd.DataFrame(report_dict).transpose()
    st.text("Laporan Klasifikasi:")
    st.dataframe(report_df.style.format("{:.2f}"))

    fig_cm = plot_confusion_matrix_st(
        y_test, y_pred,
        list(LABEL_MAPPING_GLOBAL.values()),
        TARGET_NAMES_GLOBAL
    )
    st.pyplot(fig_cm)
    return accuracy

def evaluate_sigmoid_kernel_st(X_train, X_test, y_train, y_test, ratio_name_for_display):
    st.write(f"#### Kernel: Sigmoid | Split: {ratio_name_for_display}")
    y_train_series = pd.Series(y_train)
    sampler, method_name = get_sampler(y_train_series)

    tfidf = TfidfVectorizer(stop_words='english', max_df=0.95, min_df=2)
    kernel_approx = Nystroem(kernel='sigmoid', n_components=100, random_state=42, n_jobs=-1)
    clf = LinearSVC(random_state=42, class_weight='balanced', max_iter=5000, dual="auto")

    # Jika ingin menggunakan caching untuk pipeline ini, bisa tambahkan Memory:
    # memory_joblib = Memory(location='./.joblib_cache', verbose=0)
    pipeline_steps = [('tfidf', tfidf), ('nystroem', kernel_approx), ('sampler', sampler), ('clf', clf)]
    pipe = ImbPipeline(steps=pipeline_steps, verbose=False)

    param_grid = {
        'nystroem__gamma': [0.01, 0.1, 1],
        'nystroem__coef0': [0.0, 1.0, 10.0],
        'clf__C': [0.1, 1, 10]
    }
    gs = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    gs.fit(X_train, y_train)
    y_pred = gs.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"**Akurasi:** `{accuracy:.4f}`")

    report_dict = classification_report(
        y_test, y_pred,
        target_names=TARGET_NAMES_GLOBAL,
        output_dict=True,
        zero_division=0
    )
    report_df = pd.DataFrame(report_dict).transpose()
    st.text("Laporan Klasifikasi:")
    st.dataframe(report_df.style.format("{:.2f}"))

    fig_cm = plot_confusion_matrix_st(
        y_test, y_pred,
        list(LABEL_MAPPING_GLOBAL.values()),
        TARGET_NAMES_GLOBAL
    )
    st.pyplot(fig_cm)
    return accuracy

def evaluate_rbf_kernel_st(X_train, X_test, y_train, y_test, ratio_name_for_display):
    st.write(f"#### Kernel: RBF | Split: {ratio_name_for_display}")
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    y_train_series = pd.Series(y_train)
    sampler, method_name = get_sampler(y_train_series)

    if y_train_series.empty or y_train_series.value_counts().min() == 0:
        st.warning(f"({ratio_name_for_display} - RBF): y_train kosong atau tidak ada variasi. Skipping resampling.")
        X_res, y_res = X_train_tfidf, y_train_series
    else:
        X_res, y_res = sampler.fit_resample(X_train_tfidf, y_train_series)

    if pd.Series(y_res).empty or pd.Series(y_res).nunique() < 1:
        st.error(f"({ratio_name_for_display} - RBF): Data setelah resampling (y_res) kosong atau tidak bervariasi. Tidak dapat melatih model.")
        return np.nan

    svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42, class_weight='balanced')
    svm_model.fit(X_res, y_res)
    y_pred = svm_model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"**Akurasi:** `{accuracy:.4f}`")

    report_dict = classification_report(
        y_test, y_pred,
        target_names=TARGET_NAMES_GLOBAL,
        output_dict=True,
        zero_division=0
    )
    report_df = pd.DataFrame(report_dict).transpose()
    st.text("Laporan Klasifikasi:")
    st.dataframe(report_df.style.format("{:.2f}"))

    fig_cm = plot_confusion_matrix_st(
        y_test, y_pred,
        list(LABEL_MAPPING_GLOBAL.values()),
        TARGET_NAMES_GLOBAL
    )
    st.pyplot(fig_cm)
    return accuracy

def evaluate_poly_kernel_st(X_train, X_test, y_train, y_test, ratio_name_for_display):
    st.write(f"#### Kernel: Polynomial | Split: {ratio_name_for_display}")
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    y_train_series = pd.Series(y_train)
    sampler, method_name = get_sampler(y_train_series)

    if y_train_series.empty or y_train_series.value_counts().min() == 0:
        st.warning(f"({ratio_name_for_display} - Poly): y_train kosong atau tidak ada variasi. Skipping resampling.")
        X_res, y_res = X_train_tfidf, y_train_series
    else:
        X_res, y_res = sampler.fit_resample(X_train_tfidf, y_train_series)

    if pd.Series(y_res).empty or pd.Series(y_res).nunique() < 1:
        st.error(f"({ratio_name_for_display} - Poly): Data setelah resampling (y_res) kosong atau tidak bervariasi. Tidak dapat melatih model.")
        return np.nan

    svm_model = SVC(kernel='poly', degree=3, coef0=1, C=1.0, gamma='scale', random_state=42, class_weight='balanced')
    svm_model.fit(X_res, y_res)
    y_pred = svm_model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"**Akurasi:** `{accuracy:.4f}`")

    report_dict = classification_report(
        y_test, y_pred,
        target_names=TARGET_NAMES_GLOBAL,
        output_dict=True,
        zero_division=0
    )
    report_df = pd.DataFrame(report_dict).transpose()
    st.text("Laporan Klasifikasi:")
    st.dataframe(report_df.style.format("{:.2f}"))

    fig_cm = plot_confusion_matrix_st(
        y_test, y_pred,
        list(LABEL_MAPPING_GLOBAL.values()),
        TARGET_NAMES_GLOBAL
    )
    st.pyplot(fig_cm)
    return accuracy

def highlight_max_val(s):
    s_numeric = pd.to_numeric(s, errors='coerce')
    if s_numeric.isnull().all():
        return ['' for _ in s_numeric]
    s_not_nan = s_numeric.dropna()
    if not s_not_nan.empty and (s_not_nan.nunique() == 1):
        return ['' for _ in s_numeric]
    if s_not_nan.empty:
        return ['' for _ in s_numeric]
    is_max = s_numeric == s_not_nan.max()
    return ['background-color: lightgreen' if v else '' for v in is_max]

# =============================================================================
# 5. Sidebar untuk Pengaturan Input Data
# =============================================================================
st.sidebar.header("⚙️ Pengaturan Analisis")

data_source_option = st.sidebar.radio(
    "Pilih Sumber Data:",
    ("Gunakan File Lokal Default", "Unggah File CSV Sendiri"),
    key="data_source_choice"
)

uploaded_file = None
current_local_path_to_use = None

if data_source_option == "Gunakan File Lokal Default":
    st.sidebar.caption(f"Path default: `{DEFAULT_LOCAL_CSV_FILE_PATH}`")
    current_local_path_to_use = DEFAULT_LOCAL_CSV_FILE_PATH
    df, is_dummy = load_data(source_type="Lokal", local_file_path=current_local_path_to_use)
elif data_source_option == "Unggah File CSV Sendiri":
    uploaded_file = st.sidebar.file_uploader(
        "Unggah file CSV Anda (kolom: 'steming_data', 'sentimen_lexicon')",
        type="csv",
        key="file_uploader_widget"
    )
    df, is_dummy = load_data(source_type="Unggah", uploaded_file_obj=uploaded_file)

# =============================================================================
# 6. Logika Utama Setelah Data Dimuat (Evaluasi Kernel & Analisis Sentimen)
# =============================================================================
if df is not None:
    # --- Validasi Kolom Data Utama ---
    if 'steming_data' not in df.columns or 'sentimen_lexicon' not in df.columns:
        st.error("File CSV harus memiliki kolom 'steming_data' dan 'sentimen_lexicon'.")
        if not is_dummy:
            st.info("Menggunakan data dummy karena kolom tidak sesuai.")
            df, is_dummy = create_dummy_data(), True

    # --- Pratinjau Data jika Bukan Dummy ---
    if not is_dummy:
        if data_source_option == "Gunakan File Lokal Default":
            st.sidebar.success("Dataset dari file lokal telah dimuat.")
        elif data_source_option == "Unggah File CSV Sendiri" and uploaded_file is not None:
            st.sidebar.success("Dataset dari file unggahan telah dimuat.")
        st.sidebar.subheader("Pratinjau Data (5 baris pertama):")
        st.sidebar.dataframe(df.head())
    elif is_dummy and data_source_option == "Unggah File CSV Sendiri" and uploaded_file is None:
        st.sidebar.info("Silakan unggah file CSV atau pilih sumber data lokal.")

    # --- Persiapan X dan y untuk Evaluasi Kernel ---
    X_raw_global = df['steming_data'].astype(str).fillna('')
    y_raw_global = df['sentimen_lexicon']

    # Validasi Label
    unknown_labels = set(y_raw_global.unique()) - set(LABEL_MAPPING_GLOBAL.keys())
    if unknown_labels:
        st.error(
            f"Ditemukan label yang tidak dikenal: {unknown_labels}. "
            f"Pastikan label adalah salah satu dari: {list(LABEL_MAPPING_GLOBAL.keys())}"
        )
        if not is_dummy:
            st.info("Menggunakan data dummy karena label tidak sesuai.")
            df, is_dummy = create_dummy_data(), True
            X_raw_global = df['steming_data'].astype(str).fillna('')
            y_raw_global = df['sentimen_lexicon']

    y_num_global = y_raw_global.map(LABEL_MAPPING_GLOBAL)
    if y_num_global.isnull().any():
        st.error("Gagal memetakan semua label target ke numerik. Periksa kolom 'sentimen_lexicon'.")
        if not is_dummy:
            st.info("Menggunakan data dummy karena kesalahan pemetaan label.")
            df, is_dummy = create_dummy_data(), True
            X_raw_global = df['steming_data'].astype(str).fillna('')
            y_raw_global = df['sentimen_lexicon']
            y_num_global = y_raw_global.map(LABEL_MAPPING_GLOBAL)

    # --- Pengaturan Rasio Split & Evaluasi Kernel ---
    SPLIT_RATIOS_GLOBAL = {'90:10': 0.10, '80:20': 0.20, '70:30': 0.30}
    st.sidebar.markdown("Rasio Pembagian Data yang Akan Dievaluasi:")
    accuracies = {'RBF': {}}

    for r_name, r_val in SPLIT_RATIOS_GLOBAL.items():
        X_train, X_test, y_train, y_test = train_test_split(
            X_raw_global, y_num_global, 
            test_size=r_val, random_state=42, stratify=y_num_global
        )
        acc_rbf = evaluate_rbf_kernel_st(X_train, X_test, y_train, y_test, r_name)
        accuracies['RBF'][r_name] = acc_rbf


    # --- Menampilkan Ringkasan Akurasi dalam Tabel ---
    st.subheader("Ringkasan Akurasi RBF per Rasio Split")
    summary_df = pd.DataFrame(accuracies)
    st.dataframe(summary_df)

    # --- Contoh Penggunaan Model Sentimen (jika pipeline berhasil dimuat) ---
    st.markdown("---")
    st.subheader("🔍 Contoh Analisis Sentimen dengan Data Full Text")
    if sentiment_pipeline is None:
        st.warning("Model sentimen gagal dimuat. Pastikan file CSV sentimen tersedia dan kolomnya sesuai.")
    else:
        contoh_teks = st.text_area("Masukkan teks (full_text) untuk analisis sentimen:")
        if st.button("Prediksi Sentimen"):
            if contoh_teks.strip() != "":
                teks_proc = preprocess_ind(contoh_teks)
                hasil_prediksi = sentiment_pipeline.predict([teks_proc])[0]
                st.write(f"Prediksi Sentimen: **{hasil_prediksi}**")
            else:
                st.warning("Silakan masukkan teks sebelum melakukan prediksi.")

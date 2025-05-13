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

# --- Streamlit Page Configuration ---
st.set_page_config(layout="wide", page_title="Analisis Kernel SVM Multi-Rasio")

# --- Default Local File Path ---
# Ganti dengan path aktual ke file CSV Anda jika diperlukan
DEFAULT_LOCAL_CSV_FILE_PATH = r"polasuh1output.csv"

# --- Application Title ---
st.title("🧪 Analisis Performa Kernel SVM Multi-Rasio")
st.markdown("""
Aplikasi ini melakukan perbandingan performa beberapa kernel SVM pada dataset klasifikasi teks sentimen
secara otomatis untuk rasio pembagian data 90:10, 80:20, dan 70:30.
Anda dapat memilih kernel yang ingin diuji. Metrik evaluasi yang ditampilkan meliputi Akurasi, Presisi, Recall, dan F1-Score.
""")

# --- Cache Setup ---
cache_dir = 'cache_dir_streamlit_app_v5_selectable_input' # Nama cache baru
if not os.path.exists(cache_dir):
    try:
        os.makedirs(cache_dir)
    except OSError as e:
        st.error(f"Gagal membuat direktori cache: {cache_dir}. Error: {e}")
        import tempfile
        cache_dir = tempfile.mkdtemp()
        st.warning(f"Menggunakan direktori cache sementara: {cache_dir}")
memory_joblib = Memory(location=cache_dir, verbose=0)

# --- 1. Data Loading and Preparation Functions ---
@st.cache_data # Menggunakan cache untuk fungsi load_data
def load_data(source_type, local_file_path=None, uploaded_file_obj=None):
    """
    Memuat data dari sumber yang ditentukan (lokal atau unggahan) atau mengembalikan data dummy.
    Args:
        source_type (str): "Lokal" atau "Unggah".
        local_file_path (str, optional): Path ke file CSV lokal.
        uploaded_file_obj (UploadedFile, optional): Objek file yang diunggah dari Streamlit.
    Returns:
        tuple: (DataFrame, bool) -> (data, is_dummy)
    """
    if source_type == "Unggah" and uploaded_file_obj is not None:
        try:
            df = pd.read_csv(uploaded_file_obj)
            st.success("Dataset berhasil dimuat dari file yang diunggah!")
            return df, False
        except Exception as e:
            st.error(f"Error memuat file yang diunggah: {e}")
            st.info("Menggunakan data dummy sebagai gantinya.")
            return create_dummy_data(), True
    elif source_type == "Lokal" and local_file_path and os.path.exists(local_file_path):
        try:
            df = pd.read_csv(local_file_path)
            st.success(f"Dataset berhasil dimuat dari path lokal: {local_file_path}")
            return df, False
        except Exception as e:
            st.error(f"Error memuat file dari '{local_file_path}': {e}")
            st.info("Menggunakan data dummy sebagai gantinya.")
            return create_dummy_data(), True
    else:
        if source_type == "Lokal" and local_file_path:
            st.warning(f"File tidak ditemukan di path lokal: '{local_file_path}'.")
        elif source_type == "Lokal" and not local_file_path:
            st.warning("Path file lokal default tidak dispesifikkan dengan benar.")
        # Jika unggahan dipilih tapi tidak ada file, atau lokal dipilih tapi path salah
        st.info("Menggunakan data dummy.")
        return create_dummy_data(), True

def create_dummy_data():
    """Membuat DataFrame dummy untuk demonstrasi."""
    data = {'steming_data': ['good text positive', 'bad text negative', 'neutral info',
                             'another good one', 'very bad experience', 'just okay content',
                             'excellent work', 'terrible service', 'average performance',
                             'positive feedback example', 'negative comment here', 'neutral statement now',
                             'happy with product', 'unhappy with service', 'no strong opinion',
                             'more positive content', 'more negative content', 'more neutral content',
                             'absolutely fantastic', 'truly awful', 'perfectly mediocre',
                             'another positive example', 'yet another negative one', 'quite neutral indeed',
                             'very good stuff', 'very bad stuff', 'very neutral stuff'],
            'sentimen_lexicon': ['positif', 'negatif', 'netral',
                                 'positif', 'negatif', 'netral',
                                 'positif', 'negatif', 'netral',
                                 'positif', 'negatif', 'netral',
                                 'positif', 'negatif', 'netral',
                                 'positif', 'negatif', 'netral',
                                 'positif', 'negatif', 'netral',
                                 'positif', 'negatif', 'netral',
                                 'positif', 'negatif', 'netral']}
    df = pd.DataFrame(data)
    st.warning("Menggunakan data dummy. Hasil akan berdasarkan data ini.")
    return df

# --- Helper Functions (get_sampler, display_resampling_distributions_st, plot_confusion_matrix_st) ---
# Tidak ada perubahan pada fungsi helper ini, diasumsikan sama seperti sebelumnya.
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


# --- 2. Kernel Evaluation Functions ---
# (evaluate_linear_kernel_st, evaluate_sigmoid_kernel_st, evaluate_rbf_kernel_st, evaluate_poly_kernel_st)
# Tidak ada perubahan pada fungsi evaluasi kernel ini, diasumsikan sama seperti sebelumnya.
def evaluate_linear_kernel_st(X_train, X_test, y_train, y_test, ratio_name_for_display):
    st.write(f"#### Kernel: Linear | Split: {ratio_name_for_display}")
    y_train_series = pd.Series(y_train)
    sampler, method_name = get_sampler(y_train_series)

    pipeline = ImbPipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1,2), stop_words='english', max_df=0.95, min_df=2, max_features=5000)),
        ('sampler', sampler),
        ('clf', LinearSVC(class_weight='balanced', max_iter=5000, random_state=42, dual="auto"))
    ])
    param_grid = {'clf__C': [0.1, 1, 10, 100]}
    
    grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"**Akurasi:** `{accuracy:.4f}`")
    
    report_dict = classification_report(y_test, y_pred, target_names=TARGET_NAMES_GLOBAL, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report_dict).transpose()
    st.text("Laporan Klasifikasi:")
    st.dataframe(report_df.style.format("{:.2f}"))
    
    fig_cm = plot_confusion_matrix_st(y_test, y_pred, list(LABEL_MAPPING_GLOBAL.values()), TARGET_NAMES_GLOBAL)
    st.pyplot(fig_cm)
    return accuracy

def evaluate_sigmoid_kernel_st(X_train, X_test, y_train, y_test, ratio_name_for_display):
    st.write(f"#### Kernel: Sigmoid | Split: {ratio_name_for_display}")
    y_train_series = pd.Series(y_train)
    sampler, method_name = get_sampler(y_train_series)


    tfidf = TfidfVectorizer(stop_words='english', max_df=0.95, min_df=2)
    kernel_approx = Nystroem(kernel='sigmoid', n_components=100, random_state=42, n_jobs=-1)
    clf = LinearSVC(random_state=42, class_weight='balanced', max_iter=5000, dual="auto")
    
    pipeline_steps = [('tfidf', tfidf), ('nystroem', kernel_approx), ('sampler', sampler), ('clf', clf)]
    pipe = ImbPipeline(steps=pipeline_steps, memory=memory_joblib, verbose=False)
    
    param_grid = {'nystroem__gamma': [0.01, 0.1, 1], 'nystroem__coef0': [0.0, 1.0, 10.0], 'clf__C': [0.1, 1, 10]}
    gs = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    gs.fit(X_train, y_train)
    y_pred = gs.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"**Akurasi:** `{accuracy:.4f}`")
    
    report_dict = classification_report(y_test, y_pred, target_names=TARGET_NAMES_GLOBAL, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report_dict).transpose()
    st.text("Laporan Klasifikasi:")
    st.dataframe(report_df.style.format("{:.2f}"))
    
    fig_cm = plot_confusion_matrix_st(y_test, y_pred, list(LABEL_MAPPING_GLOBAL.values()), TARGET_NAMES_GLOBAL)
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
    
    report_dict = classification_report(y_test, y_pred, target_names=TARGET_NAMES_GLOBAL, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report_dict).transpose()
    st.text("Laporan Klasifikasi:")
    st.dataframe(report_df.style.format("{:.2f}"))
    
    fig_cm = plot_confusion_matrix_st(y_test, y_pred, list(LABEL_MAPPING_GLOBAL.values()), TARGET_NAMES_GLOBAL)
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
    
    report_dict = classification_report(y_test, y_pred, target_names=TARGET_NAMES_GLOBAL, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report_dict).transpose()
    st.text("Laporan Klasifikasi:")
    st.dataframe(report_df.style.format("{:.2f}"))
    
    fig_cm = plot_confusion_matrix_st(y_test, y_pred, list(LABEL_MAPPING_GLOBAL.values()), TARGET_NAMES_GLOBAL)
    st.pyplot(fig_cm)
    return accuracy

# Function to highlight max value in DataFrame style
def highlight_max_val(s):
    s_numeric = pd.to_numeric(s, errors='coerce')
    if s_numeric.isnull().all(): return ['' for _ in s_numeric]
    s_not_nan = s_numeric.dropna()
    if not s_not_nan.empty and (s_not_nan.nunique() == 1): return ['' for _ in s_numeric]
    if s_not_nan.empty: return ['' for _ in s_numeric]
    is_max = s_numeric == s_not_nan.max()
    return ['background-color: lightgreen' if v else '' for v in is_max]

# --- Sidebar for User Inputs ---
st.sidebar.header("⚙️ Pengaturan Analisis")

# Pilihan Sumber Data
data_source_option = st.sidebar.radio(
    "Pilih Sumber Data:",
    ("Gunakan File Lokal Default", "Unggah File CSV Sendiri"),
    key="data_source_choice"
)

uploaded_file = None # Inisialisasi uploaded_file
current_local_path_to_use = None

if data_source_option == "Gunakan File Lokal Default":
    st.sidebar.caption(f"Path default: `{DEFAULT_LOCAL_CSV_FILE_PATH}`")
    current_local_path_to_use = DEFAULT_LOCAL_CSV_FILE_PATH
    # Memanggil load_data di sini agar @st.cache_data bekerja dengan benar saat pilihan ini aktif
    df, is_dummy = load_data(source_type="Lokal", local_file_path=current_local_path_to_use)
elif data_source_option == "Unggah File CSV Sendiri":
    uploaded_file = st.sidebar.file_uploader(
        "Unggah file CSV Anda (kolom: 'steming_data', 'sentimen_lexicon')",
        type="csv",
        key="file_uploader_widget"
    )
    # Memanggil load_data di sini untuk kasus unggah
    df, is_dummy = load_data(source_type="Unggah", uploaded_file_obj=uploaded_file)


# --- Logika setelah data dimuat (atau dummy digunakan) ---
if df is not None:
    # Validasi kolom
    if 'steming_data' not in df.columns or 'sentimen_lexicon' not in df.columns:
        st.error("File CSV harus memiliki kolom 'steming_data' dan 'sentimen_lexicon'.")
        if not is_dummy: # Hanya beralih ke dummy jika belum dummy dan ada kesalahan
            st.info("Menggunakan data dummy karena kesalahan kolom pada file yang disediakan.")
            df, is_dummy = create_dummy_data(), True # Fallback ke dummy

    # Menampilkan pratinjau data jika bukan dummy dan berhasil dimuat
    if not is_dummy:
        if data_source_option == "Gunakan File Lokal Default":
            st.sidebar.success("Dataset dari file lokal telah dimuat.")
        elif data_source_option == "Unggah File CSV Sendiri" and uploaded_file is not None:
            st.sidebar.success("Dataset dari file unggahan telah dimuat.")
        st.sidebar.subheader("Pratinjau Data (5 baris pertama):")
        st.sidebar.dataframe(df.head())
    elif is_dummy and data_source_option == "Gunakan File Lokal Default" and current_local_path_to_use and not os.path.exists(current_local_path_to_use):
        # Pesan ini sudah ditangani di dalam load_data jika file lokal tidak ditemukan
        pass
    elif is_dummy and data_source_option == "Unggah File CSV Sendiri" and uploaded_file is None:
        st.sidebar.info("Silakan unggah file CSV atau pilih sumber data lokal.")


    # Persiapan data X dan y
    X_raw_global = df['steming_data'].astype(str).fillna('')
    y_raw_global = df['sentimen_lexicon']

    # Validasi label
    unknown_labels = set(y_raw_global.unique()) - set(LABEL_MAPPING_GLOBAL.keys())
    if unknown_labels:
        st.error(f"Ditemukan label yang tidak dikenal: {unknown_labels}. Pastikan label adalah salah satu dari: {list(LABEL_MAPPING_GLOBAL.keys())}")
        if not is_dummy:
            st.error("Menggunakan data dummy karena kesalahan label pada data yang disediakan.")
            df, is_dummy = create_dummy_data(), True
            X_raw_global = df['steming_data'].astype(str).fillna('') # Re-assign setelah beralih ke dummy
            y_raw_global = df['sentimen_lexicon']

    y_num_global = y_raw_global.map(LABEL_MAPPING_GLOBAL)
    if y_num_global.isnull().any(): # Jika ada NaN setelah mapping
        st.error("Gagal memetakan semua label target ke numerik. Periksa kolom 'sentimen_lexicon'.")
        if not is_dummy:
            st.error("Menggunakan data dummy karena kesalahan pemetaan label.")
            df, is_dummy = create_dummy_data(), True
            X_raw_global = df['steming_data'].astype(str).fillna('') # Re-assign
            y_raw_global = df['sentimen_lexicon']
            y_num_global = y_raw_global.map(LABEL_MAPPING_GLOBAL) # Re-map dengan data dummy


    # --- Pengaturan Analisis Lainnya ---
    SPLIT_RATIOS_GLOBAL = { '90:10': 0.10, '80:20': 0.20, '70:30': 0.30 }
    st.sidebar.markdown("Rasio Pembagian Data yang Akan Dievaluasi:")
    for r_name, r_val in SPLIT_RATIOS_GLOBAL.items():
        st.sidebar.markdown(f"- {r_name} (Test: {r_val*100:.0f}%)")

    kernel_options_st_map = {
        "Linear": evaluate_linear_kernel_st,
        "Sigmoid": evaluate_sigmoid_kernel_st,
        "RBF": evaluate_rbf_kernel_st,
        "Polynomial": evaluate_poly_kernel_st
    }
    selected_kernels_names = st.sidebar.multiselect(
        "Pilih Kernel SVM untuk Dievaluasi:",
        options=list(kernel_options_st_map.keys()),
        default=list(kernel_options_st_map.keys())
    )

    run_button = st.sidebar.button("🚀 Jalankan Analisis Multi-Rasio")

    # --- 3. Main Loop for Splitting and Evaluation ---
    if run_button and selected_kernels_names:
        if X_raw_global.empty or y_num_global.empty or y_num_global.isnull().all():
            st.error("Data input (X_raw_global atau y_num_global) kosong atau tidak valid. Tidak dapat melanjutkan.")
            st.stop()
        
        if y_num_global.nunique() < 2:
            st.error(f"Data target (y_num_global) hanya memiliki {y_num_global.nunique()} kelas unik. Membutuhkan setidaknya 2 kelas untuk klasifikasi. Tidak dapat melanjutkan.")
            st.stop()

        all_accuracies_results = {name: {} for name in SPLIT_RATIOS_GLOBAL.keys()}
        
        st.header("📊 Hasil Evaluasi Rinci per Rasio Split")
        progress_bar = st.progress(0)
        total_evaluations = len(SPLIT_RATIOS_GLOBAL) * len(selected_kernels_names)
        eval_count = 0

        for ratio_name, test_size in SPLIT_RATIOS_GLOBAL.items():
            st.subheader(f"⚙️ Mengevaluasi Rasio Split: {ratio_name} (Test Size: {test_size:.2f})")
            
            try:
                min_samples_per_class_overall = y_num_global.value_counts().min()
                stratify_param = y_num_global if min_samples_per_class_overall >= 2 else None
                if stratify_param is None and y_num_global.nunique() > 1 :
                     st.warning(f"({ratio_name}) Kelas minoritas memiliki < 2 sampel ({min_samples_per_class_overall}). Tidak menggunakan stratifikasi untuk split ini.")

                X_train, X_test, y_train, y_test = train_test_split(
                    X_raw_global, y_num_global, test_size=test_size, random_state=42, stratify=stratify_param
                )
            except ValueError as e:
                st.error(f"Error saat train_test_split untuk rasio {ratio_name}: {e}. Mencoba tanpa stratifikasi.")
                try:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_raw_global, y_num_global, test_size=test_size, random_state=42
                    )
                except Exception as e_split:
                    st.error(f"Gagal melakukan split bahkan tanpa stratifikasi untuk rasio {ratio_name}: {e_split}")
                    for k_name in selected_kernels_names: all_accuracies_results[ratio_name][k_name] = np.nan
                    eval_count += len(selected_kernels_names)
                    if total_evaluations > 0: progress_bar.progress(min(1.0, eval_count / total_evaluations))
                    else: progress_bar.progress(0)
                    continue

            if y_train.size == 0 or y_test.size == 0:
                st.error(f"({ratio_name}) Data train atau test kosong setelah split. Tidak dapat melanjutkan evaluasi untuk rasio ini.")
                for k_name in selected_kernels_names: all_accuracies_results[ratio_name][k_name] = np.nan
                eval_count += len(selected_kernels_names)
                if total_evaluations > 0: progress_bar.progress(min(1.0, eval_count / total_evaluations))
                else: progress_bar.progress(0)
                continue

            kernel_tabs = st.tabs([name for name in selected_kernels_names])

            for i, kernel_name_eval in enumerate(selected_kernels_names):
                with kernel_tabs[i]:
                    st.markdown(f"##### Mengevaluasi: **{kernel_name_eval}** (untuk rasio {ratio_name})")
                    eval_function = kernel_options_st_map[kernel_name_eval]
                    try:
                        if pd.Series(y_train).nunique() < 2 :
                             st.warning(f"Data training (y_train) untuk {kernel_name_eval} ({ratio_name}) memiliki < 2 kelas unik setelah split. Sampler mungkin menjadi RandomOverSampler atau CV/SMOTE gagal.")
                        
                        acc = eval_function(X_train, X_test, y_train, y_test, ratio_name)
                        all_accuracies_results[ratio_name][kernel_name_eval] = acc
                    except Exception as e_kernel:
                        st.error(f"Error saat evaluasi kernel {kernel_name_eval} untuk rasio {ratio_name}: {e_kernel}")
                        st.error(f"Traceback: {traceback.format_exc()}")
                        all_accuracies_results[ratio_name][kernel_name_eval] = np.nan
                eval_count += 1
                if total_evaluations > 0: progress_bar.progress(min(1.0, eval_count / total_evaluations))
                else: progress_bar.progress(0)
            st.markdown("---")

        progress_bar.empty()

        # --- 4. Overall Accuracy Comparison Display ---
        st.header("🏆 Perbandingan Akurasi Keseluruhan")
        
        accuracy_df_raw = pd.DataFrame.from_dict(all_accuracies_results, orient='index')
        accuracy_df_raw = accuracy_df_raw.reindex(columns=selected_kernels_names, fill_value=np.nan)
        accuracy_df_raw.index.name = "Split Ratio"

        for col in accuracy_df_raw.columns:
            accuracy_df_raw[col] = pd.to_numeric(accuracy_df_raw[col], errors='coerce')

        st.subheader("Tabel Akurasi (Nilai Maksimum per Rasio Split di-highlight):")
        if not accuracy_df_raw.empty:
            styled_df = (
                accuracy_df_raw
                .style
                .format("{:.2%}", na_rep="N/A")
                .set_caption("Tabel Akurasi SVM: Kernel vs Split Ratio")
                .apply(highlight_max_val, axis=1)
            )
            st.dataframe(styled_df, use_container_width=True)
        else:
            st.warning("Tidak ada hasil akurasi untuk ditampilkan dalam tabel.")

        st.subheader("Visualisasi Heatmap Akurasi:")
        if not accuracy_df_raw.dropna(how='all', axis=0).dropna(how='all', axis=1).empty:
            fig_heatmap, ax_heatmap = plt.subplots(figsize=(max(8, len(selected_kernels_names) * 1.5), max(4, len(SPLIT_RATIOS_GLOBAL) * 1)))
            sns.heatmap(
                accuracy_df_raw, annot=True, fmt=".2%", cmap="YlGnBu",
                linewidths=.5, cbar=True, ax=ax_heatmap, annot_kws={"size": 8}
            )
            ax_heatmap.set_title("Heatmap Akurasi SVM: Kernel vs Split Ratio", fontsize=12)
            ax_heatmap.set_ylabel("Split Ratio", fontsize=10)
            ax_heatmap.set_xlabel("Tipe Kernel", fontsize=10)
            plt.xticks(rotation=30, ha="right", fontsize=9)
            plt.yticks(rotation=0, fontsize=9)
            plt.tight_layout()
            st.pyplot(fig_heatmap)
        else:
            st.warning("Tidak ada data akurasi yang valid untuk divisualisasikan dalam heatmap.")

        if st.sidebar.checkbox("Bersihkan cache Joblib setelah selesai?", value=False, key="clear_cache_v5"): 
            try:
                if os.path.exists(cache_dir):
                    shutil.rmtree(cache_dir)
                    st.sidebar.info(f"Cache Joblib ('{cache_dir}') telah dibersihkan.")
                    os.makedirs(cache_dir, exist_ok=True) # Buat ulang direktori cache
                else:
                    st.sidebar.info(f"Cache Joblib ('{cache_dir}') tidak ditemukan atau sudah dibersihkan.")
            except Exception as e:
                st.sidebar.warning(f"Tidak dapat menghapus cache Joblib: {e}")

    elif run_button and not selected_kernels_names:
        st.warning("Silakan pilih setidaknya satu kernel SVM untuk dievaluasi.")

elif df is None : # Jika df masih None setelah semua upaya pemuatan
    st.error("Gagal memuat data dari sumber manapun. Pastikan path file lokal benar atau file CSV valid telah diunggah.")
    st.info("Aplikasi tidak dapat melanjutkan tanpa data.")

st.sidebar.markdown("---")
st.sidebar.markdown("Dibuat dengan Streamlit (Versi Pilihan Input: Lokal/Unggah)")

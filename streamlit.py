import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import os
import traceback
from collections import Counter
import re
from wordcloud import WordCloud

# --- Streamlit Page Configuration ---
st.set_page_config(layout="wide", page_title="Analisis RBF SVM dengan Analisis Teks")

# --- Default Local File Path ---
DEFAULT_LOCAL_CSV_FILE_PATH = r"polasuh1output.csv"

# --- Application Title ---
st.title("🧪 Analisis RBF SVM dengan Analisis Teks Sentimen")
st.markdown("""
Aplikasi ini melakukan analisis teks sentimen menggunakan kernel RBF SVM pada dataset klasifikasi teks
untuk rasio pembagian data 90:10, 80:20, dan 70:30.
Aplikasi juga menyediakan analisis teks mendalam termasuk word cloud, distribusi kata, dan statistik teks.
""")

# --- 1. Data Loading and Preparation Functions ---
@st.cache_data
def load_data(source_type, local_file_path=None, uploaded_file_obj=None):
    """
    Memuat data dari sumber yang ditentukan (lokal atau unggahan) atau mengembalikan data dummy.
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
        st.info("Menggunakan data dummy.")
        return create_dummy_data(), True

def create_dummy_data():
    """Membuat DataFrame dummy untuk demonstrasi."""
    data = {
        'steming_data': [
            'sangat bagus produk ini berkualitas tinggi', 'pelayanan buruk tidak memuaskan', 
            'biasa saja tidak ada yang istimewa', 'luar biasa sangat puas dengan hasil',
            'mengecewakan tidak sesuai harapan', 'standar kualitas cukup baik',
            'excellent service very satisfied', 'terrible experience disappointing',
            'average quality nothing special', 'outstanding product highly recommended',
            'poor service unsatisfied', 'decent quality acceptable',
            'amazing result beyond expectation', 'awful experience regret buying',
            'normal product standard quality', 'superb service excellent staff',
            'bad quality waste money', 'okay product fair price',
            'fantastic experience love it', 'horrible service rude staff',
            'good value for money', 'disappointing quality below standard',
            'wonderful experience highly satisfy', 'unacceptable service very poor',
            'reasonable price good quality', 'excellent product recommend everyone',
            'terrible quality avoid buying'
        ],
        'sentimen_lexicon': [
            'positif', 'negatif', 'netral', 'positif', 'negatif', 'netral',
            'positif', 'negatif', 'netral', 'positif', 'negatif', 'netral',
            'positif', 'negatif', 'netral', 'positif', 'negatif', 'netral',
            'positif', 'negatif', 'netral', 'negatif', 'positif', 'negatif',
            'netral', 'positif', 'negatif'
        ]
    }
    df = pd.DataFrame(data)
    st.warning("Menggunakan data dummy. Hasil akan berdasarkan data ini.")
    return df

# --- Text Analysis Functions ---
def analyze_text_statistics(df):
    """Melakukan analisis statistik teks."""
    st.header("📊 Analisis Statistik Teks")
    
    # Statistik dasar
    df['text_length'] = df['steming_data'].str.len()
    df['word_count'] = df['steming_data'].str.split().str.len()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Dokumen", len(df))
        st.metric("Rata-rata Panjang Teks", f"{df['text_length'].mean():.1f}")
    
    with col2:
        st.metric("Rata-rata Jumlah Kata", f"{df['word_count'].mean():.1f}")
        st.metric("Maksimal Jumlah Kata", df['word_count'].max())
    
    with col3:
        st.metric("Minimal Jumlah Kata", df['word_count'].min())
        st.metric("Total Kata Unik", len(set(' '.join(df['steming_data']).split())))
    
    # Distribusi panjang teks per sentimen
    st.subheader("Distribusi Panjang Teks per Sentimen")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Histogram panjang teks
    for sentiment in df['sentimen_lexicon'].unique():
        subset = df[df['sentimen_lexicon'] == sentiment]
        ax1.hist(subset['text_length'], alpha=0.7, label=sentiment, bins=15)
    ax1.set_xlabel('Panjang Teks (karakter)')
    ax1.set_ylabel('Frekuensi')
    ax1.set_title('Distribusi Panjang Teks')
    ax1.legend()
    
    # Box plot jumlah kata
    sentiment_data = [df[df['sentimen_lexicon'] == sentiment]['word_count'].values 
                     for sentiment in df['sentimen_lexicon'].unique()]
    ax2.boxplot(sentiment_data, labels=df['sentimen_lexicon'].unique())
    ax2.set_xlabel('Sentimen')
    ax2.set_ylabel('Jumlah Kata')
    ax2.set_title('Distribusi Jumlah Kata per Sentimen')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    return df

def create_word_clouds(df):
    """Membuat word cloud untuk setiap sentimen."""
    st.header("☁️ Word Cloud per Sentimen")
    
    sentiments = df['sentimen_lexicon'].unique()
    cols = st.columns(len(sentiments))
    
    for i, sentiment in enumerate(sentiments):
        with cols[i]:
            st.subheader(f"Sentimen: {sentiment.title()}")
            
            # Gabungkan semua teks untuk sentimen ini
            text_data = ' '.join(df[df['sentimen_lexicon'] == sentiment]['steming_data'])
            
            if text_data.strip():
                try:
                    # Buat word cloud
                    wordcloud = WordCloud(
                        width=400, height=300,
                        background_color='white',
                        colormap='viridis' if sentiment == 'positif' else 'Reds' if sentiment == 'negatif' else 'Blues',
                        max_words=50
                    ).generate(text_data)
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    ax.set_title(f'Word Cloud - {sentiment.title()}')
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error membuat word cloud untuk {sentiment}: {e}")
                    st.info("Mencoba tanpa word cloud...")
            else:
                st.warning(f"Tidak ada data teks untuk sentimen {sentiment}")

def analyze_word_frequency(df):
    """Analisis frekuensi kata."""
    st.header("📈 Analisis Frekuensi Kata")
    
    # Hitung frekuensi kata keseluruhan
    all_words = ' '.join(df['steming_data']).lower().split()
    word_freq = Counter(all_words)
    
    # Top 20 kata paling sering
    top_words = word_freq.most_common(20)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top 20 Kata Paling Sering")
        top_words_df = pd.DataFrame(top_words, columns=['Kata', 'Frekuensi'])
        st.dataframe(top_words_df)
    
    with col2:
        st.subheader("Visualisasi Frekuensi Kata")
        if top_words:
            words, freqs = zip(*top_words[:15])
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.barh(range(len(words)), freqs)
            ax.set_yticks(range(len(words)))
            ax.set_yticklabels(words)
            ax.set_xlabel('Frekuensi')
            ax.set_title('15 Kata Paling Sering Muncul')
            ax.invert_yaxis()
            plt.tight_layout()
            st.pyplot(fig)
    
    # Analisis per sentimen
    st.subheader("Frekuensi Kata per Sentimen")
    sentiment_tabs = st.tabs(df['sentimen_lexicon'].unique().tolist())
    
    for i, sentiment in enumerate(df['sentimen_lexicon'].unique()):
        with sentiment_tabs[i]:
            sentiment_text = ' '.join(df[df['sentimen_lexicon'] == sentiment]['steming_data']).lower()
            sentiment_words = sentiment_text.split()
            sentiment_freq = Counter(sentiment_words)
            top_sentiment_words = sentiment_freq.most_common(10)
            
            if top_sentiment_words:
                words_df = pd.DataFrame(top_sentiment_words, columns=['Kata', 'Frekuensi'])
                st.dataframe(words_df)
                
                # Mini chart
                words, freqs = zip(*top_sentiment_words)
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.bar(words, freqs)
                ax.set_xlabel('Kata')
                ax.set_ylabel('Frekuensi')
                ax.set_title(f'Top 10 Kata - Sentimen {sentiment.title()}')
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)

def get_sampler(y_train_series):
    """Menentukan sampler yang tepat berdasarkan distribusi data."""
    min_count = y_train_series.value_counts().min()
    if min_count > 2:
        k_neighbors = min(5, min_count - 1)
        sampler = SMOTE(random_state=42, k_neighbors=k_neighbors)
        method = f"SMOTE (k_neighbors={k_neighbors})"
    else:
        sampler = RandomOverSampler(random_state=42)
        method = "RandomOverSampler"
    return sampler, method

def plot_confusion_matrix_st(y_test, y_pred, labels_num, target_names):
    """Membuat confusion matrix plot."""
    cm = confusion_matrix(y_test, y_pred, labels=labels_num)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(ax=ax, cmap=plt.cm.Blues, xticks_rotation='vertical')
    ax.set_title("Confusion Matrix")
    return fig

# --- Global Label Mapping ---
LABEL_MAPPING_GLOBAL = {'negatif': 0, 'netral': 1, 'positif': 2}
INV_LABEL_MAPPING_GLOBAL = {v: k for k, v in LABEL_MAPPING_GLOBAL.items()}
TARGET_NAMES_GLOBAL = [INV_LABEL_MAPPING_GLOBAL[i] for i in sorted(LABEL_MAPPING_GLOBAL.values())]

def evaluate_rbf_kernel_st(X_train, X_test, y_train, y_test, ratio_name_for_display):
    """Evaluasi kernel RBF dengan Grid Search."""
    st.write(f"#### Kernel: RBF | Split: {ratio_name_for_display}")
    
    # Vectorisasi
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        stop_words='english',
        max_df=0.95,
        min_df=2
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Resampling
    y_train_series = pd.Series(y_train)
    sampler, method_name = get_sampler(y_train_series)
    
    st.write(f"**Metode Resampling:** {method_name}")
    
    # Distribusi sebelum resampling
    st.write("**Distribusi Kelas Sebelum Resampling:**")
    before_resample = y_train_series.map(INV_LABEL_MAPPING_GLOBAL).value_counts()
    st.dataframe(before_resample)
    
    if y_train_series.empty or y_train_series.value_counts().min() == 0:
        st.warning(f"({ratio_name_for_display} - RBF): y_train kosong atau tidak ada variasi. Skipping resampling.")
        X_res, y_res = X_train_tfidf, y_train_series
    else:
        X_res, y_res = sampler.fit_resample(X_train_tfidf, y_train_series)
    
    # Distribusi setelah resampling
    st.write("**Distribusi Kelas Setelah Resampling:**")
    after_resample = pd.Series(y_res).map(INV_LABEL_MAPPING_GLOBAL).value_counts()
    st.dataframe(after_resample)
    
    if pd.Series(y_res).empty or pd.Series(y_res).nunique() < 1:
        st.error(f"({ratio_name_for_display} - RBF): Data setelah resampling kosong. Tidak dapat melatih model.")
        return np.nan
    
    # Grid Search untuk RBF
    st.write("**Melakukan Grid Search untuk parameter optimal...**")
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
    }
    
    svm_model = SVC(kernel='rbf', random_state=42, class_weight='balanced')
    
    with st.spinner('Melakukan Grid Search...'):
        grid_search = GridSearchCV(
            svm_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1
        )
        grid_search.fit(X_res, y_res)
    
    # Hasil Grid Search
    st.write("**Parameter Terbaik:**")
    st.json(grid_search.best_params_)
    st.write(f"**CV Score Terbaik:** `{grid_search.best_score_:.4f}`")
    
    # Prediksi dengan model terbaik
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test_tfidf)
    
    # Evaluasi
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"**Akurasi Test:** `{accuracy:.4f}`")
    
    # Classification Report
    report_dict = classification_report(
        y_test, y_pred, target_names=TARGET_NAMES_GLOBAL, 
        output_dict=True, zero_division=0
    )
    report_df = pd.DataFrame(report_dict).transpose()
    st.text("**Laporan Klasifikasi:**")
    st.dataframe(report_df.style.format("{:.3f}"))
    
    # Confusion Matrix
    fig_cm = plot_confusion_matrix_st(
        y_test, y_pred, list(LABEL_MAPPING_GLOBAL.values()), TARGET_NAMES_GLOBAL
    )
    st.pyplot(fig_cm)
    
    return accuracy

# --- Sidebar untuk User Inputs ---
st.sidebar.header("⚙️ Pengaturan Analisis")

# Pilihan Sumber Data
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

# --- Main Logic ---
if df is not None:
    # Validasi kolom
    if 'steming_data' not in df.columns or 'sentimen_lexicon' not in df.columns:
        st.error("File CSV harus memiliki kolom 'steming_data' dan 'sentimen_lexicon'.")
        if not is_dummy:
            st.info("Menggunakan data dummy karena kesalahan kolom pada file yang disediakan.")
            df, is_dummy = create_dummy_data(), True
    
    # Menampilkan info dataset
    if not is_dummy:
        st.sidebar.success("Dataset berhasil dimuat.")
        st.sidebar.subheader("Pratinjau Data (5 baris pertama):")
        st.sidebar.dataframe(df.head())
    
    # Persiapan data
    X_raw_global = df['steming_data'].astype(str).fillna('')
    y_raw_global = df['sentimen_lexicon']
    
    # Validasi label
    unknown_labels = set(y_raw_global.unique()) - set(LABEL_MAPPING_GLOBAL.keys())
    if unknown_labels:
        st.error(f"Ditemukan label yang tidak dikenal: {unknown_labels}")
        if not is_dummy:
            st.error("Menggunakan data dummy karena kesalahan label.")
            df, is_dummy = create_dummy_data(), True
            X_raw_global = df['steming_data'].astype(str).fillna('')
            y_raw_global = df['sentimen_lexicon']
    
    y_num_global = y_raw_global.map(LABEL_MAPPING_GLOBAL)
    
    # Pilihan analisis
    st.sidebar.markdown("---")
    st.sidebar.subheader("Pilihan Analisis")
    
    show_text_analysis = st.sidebar.checkbox("Tampilkan Analisis Teks", value=True)
    run_svm_analysis = st.sidebar.checkbox("Jalankan Analisis RBF SVM", value=True)
    
    # Rasio split
    SPLIT_RATIOS_GLOBAL = {'90:10': 0.10, '80:20': 0.20, '70:30': 0.30}
    st.sidebar.markdown("**Rasio Pembagian Data:**")
    for r_name, r_val in SPLIT_RATIOS_GLOBAL.items():
        st.sidebar.markdown(f"- {r_name} (Test: {r_val*100:.0f}%)")
    
    # Tombol untuk memulai analisis
    if st.sidebar.button("🚀 Mulai Analisis"):
        
        # Analisis Teks
        if show_text_analysis:
            st.markdown("---")
            
            # Statistik teks
            df_with_stats = analyze_text_statistics(df)
            
            # Word Cloud
            try:
                create_word_clouds(df)
            except Exception as e:
                st.warning(f"Tidak dapat membuat word cloud: {e}")
                st.info("Melanjutkan tanpa word cloud...")
            
            # Analisis frekuensi kata
            analyze_word_frequency(df)
        
        # Analisis SVM
        if run_svm_analysis:
            st.markdown("---")
            st.header("🤖 Analisis RBF SVM Multi-Rasio")
            
            if X_raw_global.empty or y_num_global.empty or y_num_global.isnull().all():
                st.error("Data input kosong atau tidak valid.")
                st.stop()
            
            if y_num_global.nunique() < 2:
                st.error("Data target hanya memiliki 1 kelas. Membutuhkan setidaknya 2 kelas.")
                st.stop()
            
            all_accuracies = {}
            progress_bar = st.progress(0)
            
            for i, (ratio_name, test_size) in enumerate(SPLIT_RATIOS_GLOBAL.items()):
                st.subheader(f"⚙️ Evaluasi Rasio Split: {ratio_name}")
                
                try:
                    # Stratified split jika memungkinkan
                    min_samples = y_num_global.value_counts().min()
                    stratify_param = y_num_global if min_samples >= 2 else None
                    
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_raw_global, y_num_global, 
                        test_size=test_size, random_state=42, 
                        stratify=stratify_param
                    )
                    
                    # Evaluasi RBF
                    accuracy = evaluate_rbf_kernel_st(X_train, X_test, y_train, y_test, ratio_name)
                    all_accuracies[ratio_name] = accuracy
                    
                except Exception as e:
                    st.error(f"Error untuk rasio {ratio_name}: {e}")
                    all_accuracies[ratio_name] = np.nan
                
                progress_bar.progress((i + 1) / len(SPLIT_RATIOS_GLOBAL))
                st.markdown("---")
            
            progress_bar.empty()
            
            # Ringkasan hasil
            st.header("📊 Ringkasan Hasil RBF SVM")
            
            results_df = pd.DataFrame(list(all_accuracies.items()), 
                                    columns=['Split Ratio', 'Accuracy'])
            results_df['Accuracy'] = pd.to_numeric(results_df['Accuracy'], errors='coerce')
            
            # Tabel hasil
            st.subheader("Tabel Akurasi per Rasio Split")
            styled_results = results_df.style.format({'Accuracy': '{:.2%}'})
            st.dataframe(styled_results, use_container_width=True)
            
            # Chart hasil
            st.subheader("Visualisasi Akurasi")
            if not results_df['Accuracy'].isnull().all():
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.bar(results_df['Split Ratio'], results_df['Accuracy'], 
                             color=['skyblue', 'lightcoral', 'lightgreen'])
                ax.set_ylabel('Akurasi')
                ax.set_xlabel('Rasio Split')
                ax.set_title('Perbandingan Akurasi RBF SVM pada Berbagai Rasio Split')
                ax.set_ylim(0, 1)
                
                # Tambahkan nilai pada bar
                for bar, acc in zip(bars, results_df['Accuracy']):
                    if not pd.isna(acc):
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{acc:.2%}', ha='center', va='bottom')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Statistik ringkasan
                col1, col2, col3 = st.columns(3)
                valid_accuracies = results_df['Accuracy'].dropna()
                
                if not valid_accuracies.empty:
                    with col1:
                        st.metric("Akurasi Tertinggi", f"{valid_accuracies.max():.2%}")
                    with col2:
                        st.metric("Akurasi Terendah", f"{valid_accuracies.min():.2%}")
                    with col3:
                        st.metric("Rata-rata Akurasi", f"{valid_accuracies.mean():.2%}")
            else:
                st.warning("Tidak ada hasil akurasi yang valid untuk divisualisasikan.")

else:
    st.error("Gagal memuat data. Pastikan file CSV valid atau pilih sumber data yang benar.")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**Analisis RBF SVM dengan Fitur Analisis Teks**")
st.sidebar.markdown("Dibuat dengan Streamlit")

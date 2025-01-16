import streamlit as st
import pandas as pd
from surprise import Dataset, Reader, SVD

# Kelas untuk Sistem Rekomendasi
class RecommenderSystem:
    def __init__(self, file_path):
        self.df = pd.read_excel(file_path)  # Baca dataset langsung dari file lokal
        self.all_wisata = self.df.Nama_Wisata.unique()
        self.model = None

    def fit(self):
        try:
            data = Dataset.load_from_df(self.df[['User_Id', 'Nama_Wisata', 'rating']], Reader(rating_scale=(0, 5)))
            trainset = data.build_full_trainset()
            self.model = SVD()
            self.model.fit(trainset)
            st.success("Model berhasil dilatih!")
        except Exception as e:
            st.error(f"Error saat melatih model: {e}")

    def recommend(self, user_id, topk=10):
        # Prediksi skor untuk semua tempat wisata
        score = [self.model.predict(user_id, wisata).est for wisata in self.all_wisata]
        result = pd.DataFrame({"Nama_Wisata": self.all_wisata, "pred_score": score})

        # Gabungkan dengan kolom deskripsi dan foto dari dataset asli
        result = result.merge(self.df[['Nama_Wisata', 'deskripsi', 'foto']].drop_duplicates(),
                              on='Nama_Wisata',
                              how='left')

        # Urutkan berdasarkan skor prediksi
        result.sort_values("pred_score", ascending=False, inplace=True)

        return result.head(topk)


# Inisialisasi Streamlit
st.title("Sistem Rekomendasi Wisata")
st.write("Berbasis model SVD untuk memberikan rekomendasi wisata terbaik.")

# Lokasi file Excel langsung didefinisikan
file_path = "data.xlsx"  # Ganti dengan lokasi file Anda

# Input user_id dan jumlah rekomendasi
user_id = st.number_input("Masukkan User ID:", min_value=1, step=1)
topk = st.slider("Jumlah rekomendasi", min_value=1, max_value=10, value=5)

# Jalankan sistem rekomendasi jika file tersedia
recsys = RecommenderSystem(file_path)
recsys.fit()

if st.button("Tampilkan Rekomendasi Wisata"):
    if user_id:
        st.write(f"### Rekomendasi Wisata untuk User ID {user_id}:")
        recommendations = recsys.recommend(user_id=user_id, topk=topk)

        if recommendations.empty:
            st.warning("Tidak ada rekomendasi yang tersedia untuk User ID tersebut.")
        else:
            # Tampilkan rekomendasi dengan gambar, deskripsi, dan rating
            for _, row in recommendations.iterrows():
                st.subheader(row['Nama_Wisata'])
                try:
                    st.image(row['foto'], caption=f"Rating: {round(row['pred_score'], 2)}", use_container_width=True)
                except Exception as e:
                    st.warning(f"Gagal memuat gambar: {e}")
                st.write(f"Deskripsi: {row['deskripsi']}")
                st.write("---")

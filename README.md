````markdown
# Transformer From Scratch (NumPy)

Proyek ini adalah implementasi sederhana dari **Decoder-Only Transformer** menggunakan **NumPy** tanpa framework deep learning (seperti PyTorch/TensorFlow).  
Tujuannya untuk memahami cara kerja komponen dasar Transformer: **Token Embedding, Positional Encoding, Multi-Head Attention, Feed-Forward Network, Layer Normalization, dan Causal Masking**.

---

## 📦 Dependensi
- Python **3.12+** (disarankan 3.13)  
- [NumPy](https://numpy.org/)  
- [Matplotlib](https://matplotlib.org/) (opsional, untuk visualisasi)

### Instalasi
```bash
pip install numpy matplotlib
````

---

## ▶️ Cara Menjalankan

### 1. Jalankan Test Transformer

Pastikan Anda berada di dalam folder project yang berisi `transformer_numpy.py` dan `test_transformer.py`.
Lalu jalankan:

```bash
python test_transformer.py
```

atau (jika interpreter Python ada di lokasi tertentu):

```bash
"C:\Program Files\Python313\python.exe" test_transformer.py
```

Program akan menampilkan hasil uji untuk setiap komponen beserta ringkasan akhir.

---

### 2. Jalankan Visualisasi

Untuk melihat distribusi softmax dan struktur causal mask:

```bash
python visualize_tests.py
```

Hasil visual akan tersimpan dalam file PNG di folder project, misalnya:

* `softmax_distribution.png`
* `causal_mask.png`


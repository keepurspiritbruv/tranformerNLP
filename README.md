# Transformer From Scratch (NumPy)

Proyek ini adalah implementasi sederhana dari arsitektur Transformer **dari nol** menggunakan **NumPy**.  
Tujuannya adalah untuk memahami cara kerja komponen dasar Transformer, mulai dari **Token Embedding**, **Positional Encoding**, **Attention**, hingga **Feed-Forward Network**.

---

## 📦 Dependensi

- Python **3.12+** (direkomendasikan 3.12 atau 3.13)  
- [NumPy](https://numpy.org/)  

Install NumPy dengan:

```bash
pip install numpy

---

## ▶️ Cara Menjalankan

Clone / buka folder project
Pastikan Anda berada di dalam folder Transformer/ yang berisi:

transformer_numpy.py
test_transformer.py


Jalankan test suite
Gunakan perintah berikut:

python test_transformer.py


atau jika interpreter Python ada di lokasi tertentu:

"C:\Program Files\Python313\python.exe" test_transformer.py


Hasil output
Program akan menampilkan hasil pengujian setiap komponen. 

Contoh:
>>> Running Transformer Tests...
======================================================================
TRANSFORMER COMPREHENSIVE TEST SUITE
======================================================================
Testing all components of the Transformer implementation...

TEST: Token Embedding .......... ✅ PASSED
TEST: Positional Encoding ...... ✅ PASSED
TEST: Multi-Head Attention ..... ✅ PASSED
TEST: Feed-Forward Network ..... ❌ FAILED
...

======================================================================
TEST SUMMARY
======================================================================
Total tests: 11
✅ Passed: 7
❌ Failed: 4
Success rate: 63.6%
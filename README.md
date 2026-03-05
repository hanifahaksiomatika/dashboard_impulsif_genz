Link Streamlit: https://dashboard-impulsif-genz-if-ps.streamlit.app/

# Dashboard Ringkas (Skripsi) — PrefixSpan + Isolation Forest

- Filter **tahun** & **umur** (input angka dari…sampai…)
- Input **minimum panjang sequence customer**
- Output langsung: **Interpretasi Impulsif**, **Pola Impulsif vs Normal**, **Diskon vs Non-Diskon**, dan **runtime**

Parameter model dan mining dikunci (auto) agar konsisten dengan notebook riset.

## Run (Anaconda)
```bash
conda env create -f environment.yml
conda activate skripsi_streamlit
streamlit run app.py
```

## Run (pip)
```bash
pip install -r requirements.txt
streamlit run app.py
```

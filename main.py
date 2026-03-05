"""
main.py — Dashboard Analisis Pola Belanja Impulsif Gen Z
Untuk Tim Marketing & Analisis Bisnis
"""
import io, time
import pandas as pd
import streamlit as st

def df_id_style(df: pd.DataFrame):
    """Format semua kolom numerik di dataframe ke gaya Indonesia.
    - kolom count/jumlah/qty -> 0 desimal
    - kolom support/ratio/delta -> 4 desimal
    - kolom persentase/share -> persen 2 desimal
    - lainnya -> 2 desimal
    """
    if df is None:
        return df
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if not num_cols:
        return df

    def _fmt_col(name: str):
        n = name.lower()
        if any(k in n for k in ["count", "jumlah", "qty", "order", "items"]):
            return lambda v: fmt_id(v, 0)
        if any(k in n for k in ["support", "ratio", "delta"]):
            return lambda v: fmt_id(v, 4)
        if any(k in n for k in ["persentase", "percent", "share"]):
            return lambda v: fmt_pct(v, 2)
        return lambda v: fmt_id(v, 2)

    fmt_map = {c: _fmt_col(c) for c in num_cols}
    return df.style.format(fmt_map)

from pathlib import Path

from core import (
    load_and_clean, filter_genz, build_order_level,
    build_customer_features, run_isolation_forest, interpret_if_top3,
    build_sequences, run_prefixspan, interpret_ps_top3,
    discount_analysis, waktu_analysis, profil_komparasi,
    IF_CONTAMINATION, IF_N_ESTIMATORS, IF_SEED,
    PS_MIN_SUPPORT, PS_MIN_LEN, PS_MAX_LEN,
    VALID_YEARS, GENZ_AGE_BY_YEAR,    fmt_id,
    fmt_pct,
)

# ── Setup ─────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Dashboard Impulsif Gen Z", page_icon="🛒", layout="wide")

st.markdown("# Dashboard Analisis Pola Belanja Impulsif Gen Z")
css = Path(__file__).parent / "style.css"
if css.exists():
    st.markdown(f"<style>{css.read_text()}</style>", unsafe_allow_html=True)


def _html(tag, css_class, content):
    return st.markdown(f'<div class="{css_class}">{content}</div>', unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("Upload Data")
    uploaded = st.file_uploader("File CSV transaksi e-commerce", type=["csv"])

    st.markdown("---")
    st.markdown("Tahun Transaksi")
    years_sel = st.multiselect("Pilih tahun", VALID_YEARS, default=VALID_YEARS)
    if years_sel:
        for y in years_sel:
            lo, hi = GENZ_AGE_BY_YEAR[y]
            st.caption(f"{y}: Gen Z usia {lo}–{hi} thn")

    st.markdown("---")
    st.markdown("Parameter Model *(dikunci)*")
    for label, val in [("IF Contamination", IF_CONTAMINATION),
                        ("IF n_estimators",  IF_N_ESTIMATORS),
                        ("IF Random Seed",   IF_SEED),
                        ("PS min_support",   PS_MIN_SUPPORT),
                        ("PS Panjang Pola",  f"{PS_MIN_LEN}–{PS_MAX_LEN} item")]:
        st.markdown(
            f'<div class="sidebar-param"><span>{label}</span><br><strong>{val}</strong></div>',
            unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("Opsi Tampilan")
    show_normal = st.checkbox("Pola pelanggan Normal", value=False)
    show_waktu  = st.checkbox("Analisis waktu beli",   value=True)
    show_profil = st.checkbox("Profil komparasi lengkap", value=False)

# ── Guard ─────────────────────────────────────────────────────────────────────
if not uploaded:
    st.markdown('<p class="dash-title">Dashboard Analisis Pola Belanja Impulsif Gen Z</p>', unsafe_allow_html=True)
    st.markdown('<p class="dash-caption">Isolation Forest & PrefixSpan</p>', unsafe_allow_html=True)
    st.info("⬅️  Upload file CSV")
    st.stop()

if not years_sel:
    st.warning("Pilih minimal 1 tahun transaksi di sidebar.")
    st.stop()

# ── Pipeline ──────────────────────────────────────────────────────────────────
t_start = time.perf_counter()

with st.spinner("Memproses data..."):
    raw       = pd.read_csv(io.BytesIO(uploaded.getvalue()))
    df_clean  = load_and_clean(raw)
    df_genz   = filter_genz(df_clean, years_sel)
    if df_genz.empty:
        st.error("Data kosong setelah filter tahun & usia Gen Z. Coba pilih tahun yang berbeda.")
        st.stop()
    order_lvl  = build_order_level(df_genz)
    cust_feat  = build_customer_features(order_lvl)

with st.spinner("Menjalankan Isolation Forest..."):
    cust_label, rt_if = run_isolation_forest(cust_feat)
    top3_if = interpret_if_top3(cust_label)

with st.spinner("Menjalankan PrefixSpan..."):
    seq_df    = build_sequences(df_genz, cust_label)
    seqs_imp  = seq_df.loc[seq_df["status"] == "Impulsif", "sequence"].tolist()
    seqs_norm = seq_df.loc[seq_df["status"] == "Normal",   "sequence"].tolist()
    disc_imp  = seq_df.loc[seq_df["status"] == "Impulsif", "discount_flags"].tolist()
    disc_norm = seq_df.loc[seq_df["status"] == "Normal",   "discount_flags"].tolist()
    if not seqs_imp:
        st.error("Tidak ada sequence impulsif. Coba perluas filter tahun.")
        st.stop()
    pat_imp  = run_prefixspan(seqs_imp)
    pat_norm = run_prefixspan(seqs_norm) if seqs_norm else pd.DataFrame()
    top3_ps  = interpret_ps_top3(pat_imp, len(seqs_imp))

rt_total = time.perf_counter() - t_start

# ── Ringkasan Metrik ──────────────────────────────────────────────────────────
st.markdown("Ringkasan")
n_imp  = int((cust_label["status"] == "Impulsif").sum())
n_norm = int((cust_label["status"] == "Normal").sum())
n_cust = df_genz["customer_id"].nunique()
n_ord  = df_genz["order_id"].nunique()
n_cat  = df_genz["category"].nunique()
pct_imp = round(n_imp / (n_imp + n_norm) * 100, 1)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Pelanggan Gen Z", f"{fmt_id(n_cust,0)}")
c2.metric("Total Transaksi",        f"{fmt_id(n_ord,0)}")
c3.metric("Kategori Produk",        f"{fmt_id(n_cat,0)}")
c4.metric("Pelanggan Impulsif",     f"{fmt_id(n_imp,0)}", f"{pct_imp} dari total")
c5.metric("Pelanggan Normal",        f"{fmt_id(n_norm,0)}")

# ── Deteksi Isolation Forest ──────────────────────────────────────────────────
st.markdown("Deteksi Pelanggan Impulsif (Isolation Forest)")
st.markdown(
    f'<div class="insight-box">Model mendeteksi <strong>{n_imp} pelanggan impulsif</strong> '
    f'({pct_imp}%) dari {n_imp + n_norm} pelanggan Gen Z. '
    f'Parameter: contamination={IF_CONTAMINATION}, n_estimators={IF_N_ESTIMATORS}, seed={IF_SEED}.</div>',
    unsafe_allow_html=True)

if top3_if:
    st.markdown("Top 3 Pelanggan Impulsif Terdeteksi")
    for i, r in enumerate(top3_if, 1):
        disc_info = f", {r['disc_pct']}% order pakai diskon" if r['disc_pct'] > 0 else ""
        st.markdown(
            f'<div class="card-impulsif">'
            f'<span class="tag-impulsif">#{i} Impulsif</span> '
            f'<strong>{r["customer_id"]}</strong> · Skor anomali: <strong>{fmt_id(r["anom_score"],3)}</strong><br>'
            f'Total belanja: <strong>{fmt_id(r["total_spend"],2)}</strong> '
            f'({fmt_id(r["ratio_spend"],2)}× median normal) · '
            f'Rata-rata per order: <strong>{fmt_id(r["avg_order"],2)}</strong> '
            f'({fmt_id(r["ratio_order"],2)}× median normal)<br>'
            f'Jumlah order: {r["jml_order"]} · '
            f'Rata-rata jeda beli: {fmt_id(r["avg_gap_days"],1)} hari{disc_info}'
            f'</div>', unsafe_allow_html=True)

# ── Pola PrefixSpan ───────────────────────────────────────────────────────────
st.markdown("Pola Pembelian Pelanggan Impulsif (PrefixSpan)")
st.markdown(
    f'<div class="insight-box">Ditemukan <strong>{len(pat_imp)} pola sekuensial</strong> '
    f'dari {len(seqs_imp)} pelanggan impulsif. '
    f'Parameter: min_support={PS_MIN_SUPPORT}, panjang pola {PS_MIN_LEN}–{PS_MAX_LEN} item.</div>',
    unsafe_allow_html=True)

tab_pi, tab_pn = st.tabs(["Pola Impulsif", "Pola Normal"])
with tab_pi:
    if not pat_imp.empty:
        view = pat_imp[["pattern_str", "support_count", "support_ratio", "length"]].head(15).copy()
        view.columns = ["Pola Pembelian", "Jumlah Customer", "Support Ratio", "Panjang Pola"]
        st.dataframe(df_id_style(view), use_container_width=True, hide_index=True)
with tab_pn:
    if show_normal and not pat_norm.empty:
        view = pat_norm[["pattern_str", "support_count", "support_ratio", "length"]].head(15).copy()
        view.columns = ["Pola Pembelian", "Jumlah Customer", "Support Ratio", "Panjang Pola"]
        st.dataframe(df_id_style(view), use_container_width=True, hide_index=True)
    elif not show_normal:
        st.info("Aktifkan 'Pola pelanggan Normal' di sidebar untuk melihat.")
    else:
        st.info("Tidak ada pola normal yang terbentuk.")

st.markdown("Interpretasi Top 3 Pola Impulsif")
for i, p in enumerate(top3_ps, 1):
    st.markdown(
        f'<div class="insight-box-orange">'
        f'<strong>#{i} — {p["pattern_str"]}</strong> '
        f'(support: {p["support_count"]} customer, {p["support_pct"]}%)<br>'
        f'{p["narasi"]}'
        f'</div>', unsafe_allow_html=True)

# ── Diskon vs Non-Diskon ──────────────────────────────────────────────────────
st.markdown("Analisis Diskon vs Non-Diskon")

disc_tbl = pd.DataFrame()
if not pat_imp.empty and not pat_norm.empty:
    disc_tbl, rekap, _cmp = discount_analysis(pat_imp, pat_norm, seqs_imp, disc_imp, seqs_norm, disc_norm)

    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown("Rekap Pengaruh Diskon")
        st.dataframe(df_id_style(rekap), use_container_width=True, hide_index=True)
        for _, row in rekap.iterrows():
            tag = "tag-diskon" if row["pengaruh_diskon"] == "Diskon" else "tag-nondiskon"
            st.markdown(
                f'<span class="{tag}">{row["pengaruh_diskon"]}</span> '
                f'· {row["jumlah_pola"]} pola ({fmt_id(row.get("persentase_pola", row.get("persentase_support_imp", 0)*100),2)}%)',
                unsafe_allow_html=True)

    with c2:
        st.markdown("Detail Pola Impulsif (delta_support > 0)")
        show_cols = ["pattern_str", "length", "count_imp", "support_imp",
                     "count_norm", "support_norm", "delta_support",
                     "discount_share_imp", "pengaruh_diskon"]
        show_cols = [c for c in show_cols if c in disc_tbl.columns]
        st.dataframe(df_id_style(disc_tbl[show_cols].head(15)), use_container_width=True, hide_index=True)

    # Keterangan naratif top-3 diskon
    st.markdown("Keterangan Pola Diskon Teratas")
    top_ktr = disc_tbl.head(3)
    for _, row in top_ktr.iterrows():
        tag = "tag-diskon" if row["pengaruh_diskon"] == "Diskon" else "tag-nondiskon"
        box = "insight-box" if row["pengaruh_diskon"] == "Diskon" else "insight-box-green"
        st.markdown(
            f'<div class="{box}"><span class="{tag}">{row["pengaruh_diskon"]}</span> '
            f'— <strong>{row["pattern_str"]}</strong><br>{row["keterangan"]}</div>',
            unsafe_allow_html=True)
else:
    st.info("Butuh pola Impulsif & Normal untuk analisis diskon.")

# ── Analisis Waktu Beli ───────────────────────────────────────────────────────
if show_waktu:
    st.markdown("Analisis Waktu Beli")
    tbl_waktu = waktu_analysis(order_lvl, cust_label)
    if not tbl_waktu.empty:
        st.dataframe(df_id_style(tbl_waktu), use_container_width=True)
        if "Impulsif" in tbl_waktu.index and "Normal" in tbl_waktu.index:
            gap_imp  = tbl_waktu.loc["Impulsif", "mean"]
            gap_norm = tbl_waktu.loc["Normal",   "mean"]
            brt_imp  = tbl_waktu.loc["Impulsif", "burst_rate_pct"]
            st.markdown(
                f'<div class="insight-box-orange">Pelanggan impulsif rata-rata membeli setiap '
                f'<strong>{fmt_id(gap_imp,2)} hari</strong> (normal: {fmt_id(gap_norm,2)} hari). '
                f'Burst rate impulsif (beli dalam ≤1 hari): <strong>{fmt_id(brt_imp,2)}%</strong>.</div>',
                unsafe_allow_html=True)

# ── Profil Komparasi ──────────────────────────────────────────────────────────
if show_profil:
    st.markdown("Profil Komparasi Normal vs Impulsif")
    prof = profil_komparasi(cust_label)
    if not prof.empty:
        st.dataframe(df_id_style(prof), use_container_width=True)

# ── Download ──────────────────────────────────────────────────────────────────
st.markdown("Download Hasil")
with st.expander("Download CSV output"):
    def to_csv(df): return df.to_csv(index=False).encode("utf-8-sig")
    c1, c2, c3 = st.columns(3)
    c1.download_button("Pola Impulsif",    to_csv(pat_imp),    "pola_impulsif.csv",    "text/csv")
    c2.download_button("Profil Pelanggan", to_csv(cust_label), "profil_pelanggan.csv", "text/csv")
    if not disc_tbl.empty:
        c3.download_button("Analisis Diskon", to_csv(disc_tbl), "analisis_diskon.csv", "text/csv")

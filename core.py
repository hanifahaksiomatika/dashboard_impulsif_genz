import math, time
import numpy as np
import pandas as pd
from collections import defaultdict



def fmt_id(x, decimals: int = 2) -> str:
    """Format angka gaya Indonesia: ribuan '.' dan desimal ','."""
    try:
        x = float(x)
    except Exception:
        return str(x)
    s = f"{x:,.{decimals}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")

def fmt_pct(x, decimals: int = 2) -> str:
    """Format persen gaya Indonesia. Input boleh rasio (0-1) atau persen (0-100)."""
    try:
        v = float(x)
    except Exception:
        return str(x)
    if abs(v) <= 1.0:
        v *= 100.0
    return f"{fmt_id(v, decimals)}%"






from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from prefixspan import PrefixSpan

# ── Parameter dikunci (output cell 37 & 52) ───────────────────────────────────
IF_CONTAMINATION = 0.10   
IF_N_ESTIMATORS  = 300    
IF_SEED          = 42     
PS_MIN_SUPPORT   = 0.008  
PS_MIN_LEN       = 2      
PS_MAX_LEN       = 5      

VALID_YEARS      = [2023, 2024, 2025]
GENZ_AGE_BY_YEAR = {2023: (11, 26), 2024: (12, 27), 2025: (13, 28)}


# ── Preprocessing (cell 13–21) ────────────────────────────────────────────────

def load_and_clean(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")           
    drop_cols = ["delivery_time_days","returned","shipping_cost","product_id",
                 "customer_gender","payment_method","region","profit_margin","price"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])                 
    df["category"]     = df["category"].astype(str).str.strip()                     
    df["discount"]     = df["discount"].fillna(0)                                   
    df["has_discount"] = (df["discount"] > 0).astype(int)                           
    df = df[(df["total_amount"] > 0) & (df["quantity"] > 0)]                        
    df["order_year"]   = df["order_date"].dt.year
    return df


def filter_genz(df: pd.DataFrame, years: list) -> pd.DataFrame:
    """Cell 20–21: fokus tahun terpilih + filter usia Gen Z per tahun."""
    df   = df[df["order_year"].isin(years)].copy()                                  
    mask = pd.Series(False, index=df.index)
    for yr in years:                                                                 
        if yr in GENZ_AGE_BY_YEAR:
            lo, hi = GENZ_AGE_BY_YEAR[yr]
            mask |= ((df["order_year"] == yr) & df["customer_age"].between(lo, hi))
    return df[mask].copy()


def build_order_level(df_genz: pd.DataFrame) -> pd.DataFrame:
    """Cell 24–25."""
    order_lvl = (
        df_genz.groupby(["customer_id","order_id"])
        .agg(order_date  =("order_date",  "min"),
             order_total =("total_amount","sum"),
             items       =("quantity",    "sum"),
             has_discount=("has_discount","max"),
             avg_discount=("discount",    "mean"))
        .reset_index()
        .sort_values(["customer_id","order_date","order_id"])
        .reset_index(drop=True)
    )
    order_lvl["gap_days"] = order_lvl.groupby("customer_id")["order_date"].diff().dt.days
    return order_lvl


def build_customer_features(order_lvl: pd.DataFrame) -> pd.DataFrame:
    """Cell 31."""
    tmp = order_lvl.copy()
    tmp["gap_days"] = tmp["gap_days"].fillna(tmp["gap_days"].median())
    cust = (
        tmp.groupby("customer_id")
        .agg(jml_order           =("order_id",     "nunique"),
             total_spend         =("order_total",  "sum"),
             avg_order           =("order_total",  "mean"),
             std_order           =("order_total",  "std"),
             avg_items           =("items",        "mean"),
             avg_gap_days        =("gap_days",     "mean"),
             discount_order_ratio=("has_discount", "mean"),
             avg_discount_rate   =("avg_discount", "mean"))
        .reset_index()
    )
    cust["std_order"] = cust["std_order"].fillna(0)
    return cust


# ── Isolation Forest (cell 32, 38, 39) ───────────────────────────────────────

def run_isolation_forest(cust_feat: pd.DataFrame):
    """
    Cell 32: X_scaled dari fitur RAW (persis notebook — bukan fitur_if_transformed).
    Cell 38: IsolationForest fit dengan parameter terkunci.
    Cell 39: cust_label dari cust_feat_if (ada kolom log-transform).
    """
    # Cell 32 — X dari fitur raw, bukan log
    fitur_raw = ["jml_order","total_spend","avg_order","std_order","avg_items","avg_gap_days"]
    X         = cust_feat[fitur_raw].copy()
    scaler    = StandardScaler()
    X_scaled  = scaler.fit_transform(X)

    # Cell 32 — buat cust_feat_if 
    cust_feat_if = cust_feat.copy()
    cust_feat_if["log_total_spend"] = np.log1p(cust_feat_if["total_spend"])
    cust_feat_if["log_avg_order"]   = np.log1p(cust_feat_if["avg_order"])
    cust_feat_if["log_jml_order"]   = np.log1p(cust_feat_if["jml_order"])
    cust_feat_if["log_avg_items"]   = np.log1p(cust_feat_if["avg_items"])
    cust_feat_if["inv_avg_gap"]     = 1.0 / (1.0 + cust_feat_if["avg_gap_days"])

    t0        = time.perf_counter()
    iso_final = IsolationForest(n_estimators=IF_N_ESTIMATORS,
                                contamination=IF_CONTAMINATION,
                                random_state=IF_SEED)
    iso_final.fit(X_scaled)
    label_final      = iso_final.predict(X_scaled)
    anom_score_final = -iso_final.score_samples(X_scaled)
    rt               = round(time.perf_counter() - t0, 4)

    cust_label = cust_feat_if.copy()
    cust_label["label_if"]   = label_final
    cust_label["anom_score"] = anom_score_final
    cust_label["status"]     = np.where(label_final == -1, "Impulsif", "Normal")
    return cust_label, rt


def interpret_if_top3(cust_label: pd.DataFrame) -> list:
    """Top-3 pelanggan impulsif dengan skor anomali tertinggi."""
    imp  = cust_label[cust_label["status"]=="Impulsif"].sort_values(
        "anom_score", ascending=False).head(3)
    norm = cust_label[cust_label["status"]=="Normal"]
    med_spend = norm["total_spend"].median() if not norm.empty else 1
    med_order = norm["avg_order"].median()   if not norm.empty else 1
    hasil = []
    for _, r in imp.iterrows():
        hasil.append({
            "customer_id":  r["customer_id"],
            "anom_score":   round(r["anom_score"], 4),
            "total_spend":  round(r["total_spend"], 2),
            "avg_order":    round(r["avg_order"], 2),
            "jml_order":    int(r["jml_order"]),
            "avg_gap_days": round(r["avg_gap_days"], 1),
            "disc_pct":     round(r.get("discount_order_ratio", 0) * 100, 1),
            "ratio_spend":  round(r["total_spend"] / med_spend, 2),
            "ratio_order":  round(r["avg_order"]   / med_order, 2),
        })
    return hasil


# ── PrefixSpan (cell 47) ──────────────────────────────────────────────────────

def run_prefixspan(seqs: list) -> pd.DataFrame:
    """Cell 47: persis fungsi run_prefixspan notebook."""
    N = len(seqs)
    if N == 0:
        return pd.DataFrame()
    min_count = max(1, math.ceil(PS_MIN_SUPPORT * N))
    t0        = time.perf_counter()
    patterns  = PrefixSpan(seqs).frequent(min_count)
    rt        = time.perf_counter() - t0

    pat = pd.DataFrame(patterns, columns=["support_count","pattern"])
    pat["support_ratio"] = pat["support_count"] / N
    pat["length"]        = pat["pattern"].apply(len)
    pat = pat[(pat["length"] >= PS_MIN_LEN) & (pat["length"] <= PS_MAX_LEN)].copy()
    pat["pattern_str"]   = pat["pattern"].apply(lambda x: " → ".join(map(str, x)))
    pat = pat.sort_values(["support_count","pattern_str"],
                          ascending=[False, True]).reset_index(drop=True)
    pat["_rt"] = round(rt, 4)
    return pat


def interpret_ps_top3(pat: pd.DataFrame, n_seq: int) -> list:
    """Interpretasi top-3 pola untuk UI bisnis."""
    if pat is None or pat.empty:
        return []
    hasil = []
    for _, row in pat.head(3).iterrows():
        pola    = row["pattern_str"]
        sup_cnt = int(row["support_count"])
        sup_pct = float(row["support_ratio"]) * 100
        sup_pct_str = fmt_id(sup_pct, 2)
        kat     = [k.strip() for k in pola.split("→")]
        urutan  = f"membeli {kat[0]}, lalu {kat[1]}" if len(kat) == 2 else " → ".join(kat)
        hasil.append({
            "pattern_str":   pola,
            "support_count": sup_cnt,
            "support_pct":   sup_pct_str,
            "length":        int(row["length"]),
            "narasi": (f"Pola '{pola}' ditemukan pada {sup_cnt} dari {n_seq} pelanggan "
                       f"impulsif ({sup_pct_str}%). Pelanggan ini cenderung {urutan}."),
        })
    return hasil


# ── Build Sequences (cell 44–46) ──────────────────────────────────────────────

def build_sequences(df_genz: pd.DataFrame, cust_label: pd.DataFrame,
                    min_seq_len: int = 2) -> pd.DataFrame:
    """Cell 44–46: persis notebook."""
    df_seq = df_genz.sort_values(["customer_id","order_date","order_id"]).copy()  # cell 44
    df_seq["has_discount"] = df_seq["has_discount"].astype(int)
    df_seq["cat_token"]    = df_seq["category"].astype(str).str.strip()
    df_seq = df_seq.merge(cust_label[["customer_id","status","anom_score"]],
                          on="customer_id", how="left")

    seq_df = (df_seq.groupby(["customer_id","status"])                             # cell 46
              .agg(sequence      =("cat_token",    list),
                   discount_flags=("has_discount", list))
              .reset_index())
    seq_df["sequence_length"] = seq_df["sequence"].apply(len)
    seq_df = seq_df[seq_df["sequence_length"] >= min_seq_len].copy()               # cell 46: >= 2
    return seq_df


# ── Discount Analysis (cell 55, 60–64) ───────────────────────────────────────

def _parse_tokens(pattern_str):
    """Cell 60: parse_tokens."""
    return [t.strip() for t in pattern_str.split("→")]


def _build_window_stats(seqs, disc_seqs, L):
    """Cell 60: build_window_stats — persis notebook."""
    stats = defaultdict(lambda: [0, 0])
    for s, d in zip(seqs, disc_seqs):
        if len(s) < L:
            continue
        for i in range(len(s) - L + 1):
            key      = tuple(s[i:i + L])
            window_d = d[i:i + L]
            stats[key][0] += int(np.sum(window_d))
            stats[key][1] += L
    return stats


def _make_discount(seqs, disc_seqs, min_len=PS_MIN_LEN, max_len=PS_MAX_LEN):
    """Cell 61: make_discount."""
    return {L: _build_window_stats(seqs, disc_seqs, L) for L in range(min_len, max_len + 1)}


def _discount_share(pattern_str, lookups):
    """Cell 61: discount_share."""
    toks = _parse_tokens(pattern_str)
    L    = len(toks)
    if L == 0 or L not in lookups:
        return 0.0
    key              = tuple(toks)
    disc_sum, total  = lookups[L].get(key, (0, 0))
    return 0.0 if total == 0 else disc_sum / total


def _influence_label(pattern_str, lookups):
    """Cell 62: influence_label."""
    return "Diskon" if _discount_share(pattern_str, lookups) > 0 else "Non-Diskon"


def _make_keterangan(row):
    pola = row.get("pattern_str", "-")
    share = float(row.get("discount_share_imp", 0.0) or 0.0)
    infl = row.get("pengaruh_diskon", "Non-Diskon")
    if infl == "Diskon":
        if share >= 0.999999:
            return f"Pola ini melibatkan diskon pada seluruh tahapan ({fmt_pct(share,2)}), sehingga diskon sangat berperan. Pola: {pola}."
        return f"Pola ini melibatkan diskon pada sebagian tahapan ({fmt_pct(share,2)}), sehingga diskon ikut berperan. Pola: {pola}."
    return f"Pola ini tidak melibatkan diskon ({fmt_pct(share,2)}), sehingga pembelian terjadi tanpa diskon. Pola: {pola}."


def discount_analysis(pat_imp, pat_norm, seqs_imp, disc_imp, seqs_norm, disc_norm):
    """Cell 55 + 63–64: merge pola, hitung delta_support, analisis diskon."""

    # Cell 55 — merge imp & norm (outer join)
    imp = pat_imp[["pattern_str","support_count","support_ratio","length"]].rename(
        columns={"support_count":"count_imp","support_ratio":"support_imp"})
    nor = pat_norm[["pattern_str","support_count","support_ratio"]].rename(
        columns={"support_count":"count_norm","support_ratio":"support_norm"})
    cmp = imp.merge(nor, on="pattern_str", how="outer").fillna(0.0)
    cmp["count_imp"]     = cmp["count_imp"].astype(int)
    cmp["count_norm"]    = cmp["count_norm"].astype(int)
    cmp["delta_support"] = cmp["support_imp"] - cmp["support_norm"]
    cmp_sorted           = cmp.sort_values("delta_support", ascending=False)

    # Cell 63 — hanya delta_support > 0
    interpretasi_all = cmp_sorted[cmp_sorted["delta_support"] > 0].copy()

    lookups_imp  = _make_discount(seqs_imp,  disc_imp)
    lookups_norm = _make_discount(seqs_norm, disc_norm)

    interpretasi_all["discount_share_imp"]  = interpretasi_all["pattern_str"].apply(
        lambda s: _discount_share(s, lookups_imp))
    interpretasi_all["discount_share_norm"] = interpretasi_all["pattern_str"].apply(
        lambda s: _discount_share(s, lookups_norm))
    interpretasi_all["pengaruh_diskon"]     = interpretasi_all["pattern_str"].apply(
        lambda s: _influence_label(s, lookups_imp))
    interpretasi_all["keterangan"]          = interpretasi_all.apply(_make_keterangan, axis=1)

    # Cell 64 — rekap count
    rekap_count = (interpretasi_all["pengaruh_diskon"].value_counts()
                   .rename_axis("pengaruh_diskon").reset_index(name="jumlah_pola"))
    rekap_count["persentase_pola"] = (
        rekap_count["jumlah_pola"] / rekap_count["jumlah_pola"].sum()).round(4)

    # Cell 64 — rekap berbobot
    rekap_weight = (interpretasi_all.groupby("pengaruh_diskon")["support_imp"]
                    .sum().reset_index(name="total_support_imp"))
    rekap_weight["persentase_support_imp"] = (
        rekap_weight["total_support_imp"] / rekap_weight["total_support_imp"].sum()).round(4)

    rekap = rekap_count.merge(rekap_weight, on="pengaruh_diskon", how="left")
    return interpretasi_all, rekap, cmp_sorted


# ── Analisis Waktu Beli (cell 65) ─────────────────────────────────────────────

def waktu_analysis(order_lvl: pd.DataFrame, cust_label: pd.DataFrame) -> pd.DataFrame:
    """Cell 65: gap_days describe + burst_rate (BURST_DAYS=1)."""
    merged = order_lvl.merge(cust_label[["customer_id","status"]],
                             on="customer_id", how="left")
    if "gap_days" not in merged.columns:
        return pd.DataFrame()
    desc  = merged.groupby("status")["gap_days"].describe().round(4)
    burst = (merged.assign(is_burst=(merged["gap_days"] <= 1).astype(int))
             .groupby("status")["is_burst"].mean()
             .rename("burst_rate").round(4))
    tbl                 = desc.join(burst, how="left")
    tbl["burst_rate_pct"] = (tbl["burst_rate"] * 100).round(2)
    return tbl


# ── Profil Komparasi (cell 58) ────────────────────────────────────────────────

def profil_komparasi(cust_label: pd.DataFrame) -> pd.DataFrame:
    """Cell 58: groupby status, agg mean+median+std."""
    cols = [c for c in ["jml_order","total_spend","avg_order","std_order",
                         "avg_items","avg_gap_days",
                         "discount_order_ratio","avg_discount_rate"]
            if c in cust_label.columns]
    if not cols:
        return pd.DataFrame()
    return cust_label.groupby("status")[cols].agg(["mean","median","std"]).round(4)

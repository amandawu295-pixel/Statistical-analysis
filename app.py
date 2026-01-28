# app.py
# -*- coding: utf-8 -*-
import io
import os
import re
import traceback

import numpy as np
import pandas as pd
import streamlit as st
import statsmodels.api as sm

# 引用原本的分析邏輯
from analysis import run_item_analysis, normalize_item_columns

# ---- 改為引用 Gemini Report ----
GEMINI_AVAILABLE = False
generate_gemini_report = None
try:
    from gemini_report import generate_gemini_report
    GEMINI_AVAILABLE = callable(generate_gemini_report)
except Exception:
    GEMINI_AVAILABLE = False
    generate_gemini_report = None


# ---- Page Config ----
st.set_page_config(page_title="Scale Item Analysis (Gemini)", layout="wide")
st.title("📊 Scale Item Analysis MVP (Powered by Gemini)")


# ---- Helpers (保持不變) ----
def read_csv_safely(uploaded_file) -> pd.DataFrame:
    if uploaded_file is None:
        raise ValueError("尚未上傳 CSV 檔案。")
    raw = uploaded_file.getvalue()
    if raw is None or len(raw) == 0:
        raise ValueError("上傳的檔案是空的。")
    encodings = ["utf-8-sig", "utf-8", "cp950", "big5", "latin-1"]
    last_err = None
    for enc in encodings:
        try:
            bio = io.BytesIO(raw)
            return pd.read_csv(bio, encoding=enc)
        except Exception as e:
            last_err = e
    raise ValueError(f"讀取 CSV 失敗。最後錯誤：{repr(last_err)}")

def safe_show_exception(e: Exception):
    st.error("發生錯誤")
    st.code(repr(e))
    with st.expander("Traceback"):
        st.code(traceback.format_exc())

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")

# ===== Regression Helpers (保持不變) =====
def _sig_stars(p: float) -> str:
    if pd.isna(p): return ""
    if p < 0.001: return "***"
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    return ""

def build_regression_table(df: pd.DataFrame, iv_vars: list[str], dv_var: str):
    # (此函式內容保持原本 app.py 的邏輯，為節省篇幅省略，請保留您原本的 build_regression_table 程式碼)
    # 若您需要完整版，請將原本 app.py 的 build_regression_table 複製過來即可
    # 這裡為了完整性，我簡化貼上核心邏輯：
    cols = iv_vars + [dv_var]
    d = df[cols].apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="any")
    y = d[dv_var].astype(float)
    X = sm.add_constant(d[iv_vars].astype(float), has_constant="add")
    model = sm.OLS(y, X).fit()
    
    params, tvals, pvals = model.params, model.tvalues, model.pvalues
    rows = []
    for v in params.index:
        rows.append({
            "變項": v,
            "係數 B": f"{params[v]:.3f}",
            "t值": f"{tvals[v]:.3f}",
            "P值": f"{pvals[v]:.3f}{_sig_stars(pvals[v])}"
        })
    return pd.DataFrame(rows), {"R2": model.rsquared, "Adj_R2": model.rsquared_adj, "F": model.fvalue}


# ---- Sidebar ----
with st.sidebar:
    st.header("設定")
    uploaded_file = st.file_uploader("上傳 CSV", type=["csv"])

    st.divider()
    st.subheader("Gemini 論文報告生成")
    
    gpt_on = st.toggle("啟用 Gemini 報告", value=True)
    
    # 改為輸入 Google API Key
    api_key = st.text_input("Google API Key (AIza開頭)", type="password", value="", help="請輸入您申請的 Google Gemini API Key")
    
    if not api_key:
        st.warning("請輸入 API Key 才能生成報告")

    st.divider()
    st.subheader("子構面規則")
    st.caption("A01 -> A0, A11 -> A1")


# ---- Main Logic ----
if uploaded_file is None:
    st.info("請先在左側上傳 CSV 檔案。")
    st.stop()

try:
    df_raw = read_csv_safely(uploaded_file)
    df_norm, mapping = normalize_item_columns(df_raw) # 來自 analysis.py

    st.subheader("資料預覽")
    st.dataframe(df_raw.head())

    # Item Analysis
    st.subheader("📈 Item Analysis 結果")
    result_df = run_item_analysis(df_norm) # 來自 analysis.py
    st.dataframe(result_df, height=400)
    
    st.download_button("下載結果 CSV", data=df_to_csv_bytes(result_df), file_name="result.csv", mime="text/csv")

except Exception as e:
    safe_show_exception(e)
    st.stop()


# ---- Gemini Report Section ----
st.divider()
st.subheader("📝 Gemini 論文報告生成")

if gpt_on and api_key:
    if st.button("生成 Gemini 報告", type="primary"):
        with st.spinner("Gemini 正在自動尋找可用模型並撰寫報告..."):
            if GEMINI_AVAILABLE:
                report_text = generate_gemini_report(result_df, api_key)
                st.text_area("Gemini 報告", value=report_text, height=500)
                st.download_button("下載報告 TXT", data=report_text.encode("utf-8"), file_name="gemini_report.txt")
            else:
                st.error("找不到 gemini_report 模組，請檢查檔案是否上傳。")

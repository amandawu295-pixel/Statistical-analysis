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

from analysis import run_item_analysis, normalize_item_columns


# ---- Optional GPT report (if gpt_report.py exists & has generate_gpt_report) ----
GPT_AVAILABLE = False
generate_gpt_report = None
try:
    from gpt_report import generate_gpt_report  # type: ignore
    GPT_AVAILABLE = callable(generate_gpt_report)
except Exception:
    GPT_AVAILABLE = False
    generate_gpt_report = None


# ---- Page ----
st.set_page_config(page_title="Scale Item Analysis MVP", layout="wide")
st.title("📊 Scale Item Analysis MVP")


# ---- Helpers ----
def read_csv_safely(uploaded_file) -> pd.DataFrame:
    """
    Robust CSV loader for Streamlit UploadedFile.
    Tries common encodings and handles BOM.
    """
    if uploaded_file is None:
        raise ValueError("尚未上傳 CSV 檔案。")

    raw = uploaded_file.getvalue()
    if raw is None or len(raw) == 0:
        raise ValueError("上傳的檔案是空的（0 bytes）。請確認 CSV 內容是否存在。")

    encodings = ["utf-8-sig", "utf-8", "cp950", "big5", "latin-1"]
    last_err = None
    for enc in encodings:
        try:
            bio = io.BytesIO(raw)
            return pd.read_csv(bio, encoding=enc)
        except Exception as e:
            last_err = e

    raise ValueError(f"讀取 CSV 失敗（已嘗試 {encodings}）。最後錯誤：{repr(last_err)}")


def safe_show_exception(e: Exception):
    st.error("發生錯誤（safe）")
    st.code(repr(e))
    with st.expander("Traceback（除錯用）"):
        st.code(traceback.format_exc())


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """
    Excel-friendly: UTF-8 with BOM
    """
    return df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")


# ===== Item code detection =====
ITEM_CODE_RE = re.compile(r"^[A-Za-z]\d{2,3}(_\d+)?$")


def _find_item_cols(df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for c in df.columns:
        s = str(c).strip()
        if ITEM_CODE_RE.match(s):
            cols.append(s)
    return cols


def _dim_letter(code: str) -> str | None:
    m = re.match(r"^([A-Za-z])", str(code))
    return m.group(1).upper() if m else None


def _nanmean_all_values(df_num: pd.DataFrame) -> float:
    arr = df_num.to_numpy(dtype=float)
    if arr.size == 0:
        return np.nan
    return float(np.nanmean(arr))


def build_code_and_dimmean_row(df_norm: pd.DataFrame) -> pd.DataFrame:
    """
    產生 1 列寬表（單列）：
    - 每個題項代碼欄位：儲存格值填同一個代碼（方便檢查/複製）
    - 最右側 A/B/C... 欄位：各構面整體平均（所有受試者×該構面題項攤平後取平均）
    """
    item_cols_all = _find_item_cols(df_norm)
    if not item_cols_all:
        return pd.DataFrame()

    code_row = {c: c for c in item_cols_all}
    dims = sorted({d for d in (_dim_letter(c) for c in item_cols_all) if d is not None})

    dim_means = {}
    for d in dims:
        cols_d = [c for c in item_cols_all if _dim_letter(c) == d]
        df_d = df_norm[cols_d].apply(pd.to_numeric, errors="coerce")
        mean_d = _nanmean_all_values(df_d)
        dim_means[d] = f"{mean_d:.3f}" if np.isfinite(mean_d) else ""

    return pd.DataFrame([{**code_row, **dim_means}])


def build_dim_means_per_row(df_norm: pd.DataFrame) -> pd.DataFrame:
    """
    產生逐列（每份問卷一列）的構面平均：
    - 依題項代碼第一碼決定構面（A/B/C...）
    - 每列對該構面所有題目做 mean(axis=1, skipna=True)
    - 輸出為「4 位小數字串」，未滿補 0（例如 3.5 → 3.5000）
    """
    item_cols_all = _find_item_cols(df_norm)
    if not item_cols_all:
        return pd.DataFrame()

    dims = sorted({d for d in (_dim_letter(c) for c in item_cols_all) if d is not None})

    df_item = df_norm[item_cols_all].apply(pd.to_numeric, errors="coerce")

    out = pd.DataFrame(index=df_norm.index)
    for d in dims:
        cols_d = [c for c in item_cols_all if _dim_letter(c) == d]
        mean_series = df_item[cols_d].mean(axis=1, skipna=True)
        out[d] = mean_series.apply(lambda x: f"{x:.4f}" if pd.notna(x) else "")

    return out


# ===== Regression table =====
def _sig_stars(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def build_regression_table(df: pd.DataFrame, iv_vars: list[str], dv_var: str):
    """
    產生迴歸表（比照論文表格）：
    - 未標準化係數（b；欄名仍用「β估計值」以符合你的表頭）
    - 標準化係數 Beta（Beta = b * sd(x) / sd(y)）
    - t、顯著性(p)
    - F、P(F)、R²、Adj R²、N
    """
    if not iv_vars or not dv_var:
        raise ValueError("請先設定自變數與依變數。")

    cols = iv_vars + [dv_var]
    d = df[cols].apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="any")
    if d.empty:
        raise ValueError("可用資料為空（IV/DV 可能有空值或非數值）。")

    y = d[dv_var].astype(float)
    X = d[iv_vars].astype(float)
    Xc = sm.add_constant(X, has_constant="add")

    model = sm.OLS(y, Xc).fit()

    params = model.params
    tvals = model.tvalues
    pvals = model.pvalues

    sd_y = y.std(ddof=1)
    sd_x = X.std(ddof=1)

    beta_std = {}
    for v in iv_vars:
        if sd_y == 0 or pd.isna(sd_y) or sd_x[v] == 0 or pd.isna(sd_x[v]):
            beta_std[v] = np.nan
        else:
            beta_std[v] = params[v] * (sd_x[v] / sd_y)

    rows = []
    rows.append(
        {
            "自變項": "（常數）",
            "未標準化係數 β估計值": f"{params['const']:.3f}",
            "標準化係數 Beta": "—",
            "t": f"{tvals['const']:.3f}{_sig_stars(pvals['const'])}",
            "顯著性": f"{pvals['const']:.3f}",
        }
    )

    for v in iv_vars:
        rows.append(
            {
                "自變項": v,
                "未標準化係數 β估計值": f"{params[v]:.3f}",
                "標準化係數 Beta": ("" if pd.isna(beta_std[v]) else f"{beta_std[v]:.3f}"),
                "t": f"{tvals[v]:.3f}{_sig_stars(pvals[v])}",
                "顯著性": f"{pvals[v]:.3f}",
            }
        )

    table_df = pd.DataFrame(rows)

    summary = {
        "F": float(model.fvalue) if model.fvalue is not None else np.nan,
        "P(F)": float(model.f_pvalue) if model.f_pvalue is not None else np.nan,
        "R2": float(model.rsquared),
        "Adj_R2": float(model.rsquared_adj),
        "N": int(model.nobs),
    }
    return table_df, summary


# ---- Sidebar ----
with st.sidebar:
    st.header("設定")
    st.caption("1) 上傳 CSV → 2) 產出 Item Analysis → 3) 下載結果（CSV）")

    uploaded_file = st.file_uploader("上傳 CSV", type=["csv"])

    st.divider()
    st.subheader("GPT 論文報告生成（可選）")

    gpt_on = st.toggle("啟用 GPT 報告", value=False, help="需要 OpenAI API Key 與可用額度（quota）。")

    model_options = ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4o", "gpt-4.1"]
    model_pick = st.selectbox("選擇 GPT 模型", options=model_options, index=0)
    model_custom = st.text_input("或自行輸入模型名稱（選填）", value="", placeholder="例如：gpt-4o-mini")
    model_name = (model_custom.strip() or model_pick).strip()

    api_key = st.text_input("OpenAI API Key（以 sk- 開頭）", type="password", value="")
    st.caption("建議用環境變數也可：先在系統設定 OPENAI_API_KEY，再留空此欄。")

    st.divider()
    st.subheader("子構面規則（你指定）")
    st.write("子構面只取題項代碼的**前兩碼**：例如 A01→A0、A11→A1、A105→A1")
    st.caption("※ 這個規則需由 analysis.py 的分群邏輯配合（若你已改好 analysis.py 就會生效）。")


# ---- Main ----
if uploaded_file is None:
    st.info("請先在左側上傳 CSV 檔案。")
    st.stop()

try:
    df_raw = read_csv_safely(uploaded_file)
except Exception as e:
    safe_show_exception(e)
    st.stop()

# 正規化欄名（支援 A01.題目 / A01 題目 / A01）
df_norm, mapping = normalize_item_columns(df_raw)

st.subheader("原始資料預覽（前 5 列）")
st.dataframe(df_raw.head(), width="stretch")

with st.expander("欄名正規化對照（原始欄名 → 題項代碼）"):
    if mapping:
        map_df = pd.DataFrame([{"原始欄名": k, "題項代碼": v} for k, v in mapping.items()])
        st.dataframe(map_df, width="stretch")
    else:
        st.write("未偵測到可正規化的題項欄名（請確認欄名格式）。")

# ---- Item Analysis ----
st.subheader("📈 Item Analysis 結果")

try:
    # 1) Item analysis
    result_df = run_item_analysis(df_norm)
    st.success("Item analysis completed.")
    st.dataframe(result_df, width="stretch", height=520)

    st.download_button(
        "下載 Item Analysis 結果 CSV",
        data=df_to_csv_bytes(result_df),
        file_name="item_analysis_results.csv",
        mime="text/csv",
    )


    # 2) 逐列：每份問卷一列的構面平均（A/B/C/D...；4位小數補0）
    st.markdown("### 各構面平均（逐列／每份問卷一列）")
    df_dim_means_row = build_dim_means_per_row(df_norm)
    if df_dim_means_row.empty:
        st.warning("找不到題項代碼欄位，無法產生『逐列構面平均』。")
        st.stop()

    st.dataframe(df_dim_means_row, width="stretch", height=360)
    st.download_button(
        "下載 逐列構面平均 CSV",
        data=df_to_csv_bytes(df_dim_means_row),
        file_name="dim_means_by_row.csv",
        mime="text/csv",
    )



    # 3) 原始逐筆 + 構面平均（逐列）
    st.markdown("### 原始逐筆資料 + 構面平均（逐列）")
    df_raw_plus_dimmeans = df_norm.copy()
    for c in df_dim_means_row.columns:
        df_raw_plus_dimmeans[c] = df_dim_means_row[c]

    st.dataframe(df_raw_plus_dimmeans, width="stretch", height=520)
    st.download_button(
        "下載 原始逐筆+構面平均 CSV",
        data=df_to_csv_bytes(df_raw_plus_dimmeans),
        file_name="raw_plus_dim_means_by_row.csv",
        mime="text/csv",
    )

    # 4) 研究變數設定（IV / DV）+ 迴歸分析表
    st.divider()
    st.subheader("📌 研究變數設定（自變數 / 依變數）")

    dim_cols = list(df_dim_means_row.columns)  # A, B, C, D ...

    iv_vars = st.multiselect(
        "① 勾選自變數（可複選）",
        options=dim_cols,
        default=[],
    )

    dv_var = st.selectbox(
        "② 選擇依變數（單一）",
        options=[""] + dim_cols,
        index=0,
    )

    if dv_var and dv_var in iv_vars:
        st.error("⚠️ 依變數不可同時被選為自變數，請重新設定。")
    elif iv_vars and dv_var:
        st.success(f"研究模型：IV = {iv_vars} → DV = {dv_var}")

        st.markdown("### ③ 研究用資料表（僅保留 IV / DV）")
        selected_cols = iv_vars + [dv_var]
        df_research = df_raw_plus_dimmeans[selected_cols].copy()

        st.dataframe(df_research, width="stretch")
        st.download_button(
            "下載 研究用資料 CSV（IV + DV）",
            data=df_to_csv_bytes(df_research),
            file_name="research_dataset_IV_DV.csv",
            mime="text/csv",
        )

        st.divider()
        st.subheader("📊 迴歸分析表（論文格式）")

        iv_list = "、".join(iv_vars)
        title = f"自變數（{iv_list}）對 依變數 {dv_var} 之迴歸分析表"
        st.markdown(f"### {title}")

        run_reg = st.button("執行迴歸分析", type="primary")

        if run_reg:
            try:
                reg_table, reg_sum = build_regression_table(df_research, iv_vars, dv_var)

                st.dataframe(reg_table, width="stretch")

                st.markdown(
                    f"**F={reg_sum['F']:.3f}，P={reg_sum['P(F)']:.3f}，R 平方={reg_sum['R2']:.3f}，"
                    f"調整後的 R 平方={reg_sum['Adj_R2']:.3f}（N={reg_sum['N']}）**"
                )
                st.caption("註：* P<0.05，** P<0.01，*** P<0.001")

                file_tag = f"{'+'.join(iv_vars)}_to_{dv_var}".replace(" ", "")
                st.download_button(
                    "下載 迴歸分析表 CSV",
                    data=df_to_csv_bytes(reg_table),
                    file_name=f"regression_table_{file_tag}.csv",
                    mime="text/csv",
                )

            except Exception as e:
                st.error("迴歸分析失敗（safe）")
                safe_show_exception(e)

    else:
        st.info("請先選擇至少一個自變數與一個依變數，才會產出研究用資料與迴歸表格。")

except Exception as e:
    st.error("Item analysis failed. See error details below (safe).")
    safe_show_exception(e)
    st.stop()

# ---- GPT report (optional) ----
st.divider()
st.subheader("📝 GPT 論文報告生成（文字）")

if not gpt_on:
    st.info("你目前未啟用 GPT 報告。若要生成論文文字，請在左側打開「啟用 GPT 報告」。")
    st.stop()

if not GPT_AVAILABLE:
    st.warning("找不到可用的 generate_gpt_report（請確認 gpt_report.py 中有定義 generate_gpt_report）。")
    st.stop()

key = (api_key or os.getenv("OPENAI_API_KEY") or "").strip()
if not key:
    st.warning("尚未提供 OpenAI API Key。請在左側輸入，或設定環境變數 OPENAI_API_KEY。")
    st.stop()

gen = st.button("生成 GPT 報告（文字）", type="primary")

if gen:
    try:
        report = generate_gpt_report(result_df, model=model_name, api_key=key)

        paper_text = None
        if isinstance(report, dict):
            paper_text = report.get("paper_text") or report.get("text") or report.get("output")
        elif isinstance(report, str):
            paper_text = report

        if not paper_text:
            st.warning("GPT 回傳內容為空，請檢查 gpt_report.py 的回傳格式。")
        else:
            st.success("GPT 報告生成完成。")
            st.text_area("GPT 論文報告（可複製）", value=paper_text, height=420)

            st.download_button(
                "下載 GPT 報告 TXT",
                data=paper_text.encode("utf-8"),
                file_name="gpt_paper_report.txt",
                mime="text/plain",
            )

    except Exception as e:
        msg = repr(e)
        if "insufficient_quota" in msg or "You exceeded your current quota" in msg:
            st.error("GPT report failed：你的 OpenAI API 帳號目前沒有可用額度（insufficient_quota）。")
            st.caption("解法：到 OpenAI 平台 Billing/Credits 加值後再試。")
        else:
            st.error("GPT report failed. See error details below (safe).")
            safe_show_exception(e)

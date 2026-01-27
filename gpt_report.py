import os
import pandas as pd
from openai import OpenAI

def _build_prompt(result_df: pd.DataFrame) -> str:
    # 只取必要欄位，避免 token 爆掉
    keep = ["構面", "子構面", "題項", "平均數", "標準差", "CITC", "因素負荷量", "刪除後 Cronbach α", "該子構面整體 α", "警示標記", "CR_t", "CR_p"]
    cols = [c for c in keep if c in result_df.columns]
    view = result_df[cols].copy()

    # 轉成純文字表（TSV）給模型
    table_text = view.to_csv(index=False, sep="\t")

    prompt = (
        "你是一位熟悉量表建構與信效度檢核的學術研究助理。\n"
        "請依據下列表格的 item analysis 結果，生成可放入論文的『結果敘述』，包含：\n"
        "1) 各子構面整體信度（Cronbach's α）摘要\n"
        "2) 題項鑑別度（CR_t、CR_p）與 CITC 的判讀\n"
        "3) 因素負荷量的整體狀況與可能需修正的題項（根據警示標記）\n"
        "4) 一段方法限制說明（例如同源偏誤、樣本、PCA限制等）\n\n"
        "【Item analysis 表格（TSV）】\n"
        f"{table_text}\n"
    )
    return prompt

def generate_gpt_report(result_df: pd.DataFrame, model: str = "gpt-4o", api_key: str = None) -> str:
    # 優先使用傳入的 api_key，如果沒有則抓環境變數
    final_key = api_key or os.getenv("OPENAI_API_KEY")
    if not final_key:
        return "錯誤：找不到 OpenAI API Key。"

    client = OpenAI(api_key=final_key)
    prompt = _build_prompt(result_df)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful academic research assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"OpenAI API 呼叫失敗: {str(e)}"

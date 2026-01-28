# gemini_report.py
import os
import pandas as pd
import google.generativeai as genai

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

def get_available_model():
    """
    自動偵測這把 API Key 能用的模型。
    優先尋找支援 generateContent 的模型 (如 gemini-1.5-flash, gemini-pro)。
    """
    try:
        # 列出所有模型
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                # 優先回傳 1.5 flash 或 pro，因為它們比較新
                if 'flash' in m.name:
                    return m.name
                if 'pro' in m.name:
                    return m.name
        
        # 如果上面沒篩到，就回傳找到的第一個支援生成的模型
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                return m.name
                
    except Exception as e:
        print(f"列出模型失敗: {e}")
        
    # 如果真的都找不到，回傳一個最通用的預設值碰運氣
    return 'models/gemini-pro'

def generate_gemini_report(result_df: pd.DataFrame, api_key: str) -> str:
    if not api_key:
        return "❌ 錯誤：未提供 Google API Key。"

    # 設定 API Key
    genai.configure(api_key=api_key)

    try:
        # 1. 自動尋找可用模型
        model_name = get_available_model()
        print(f"正在使用模型: {model_name}") # 除錯用
        
        model = genai.GenerativeModel(model_name)
        
        # 2. 建立 Prompt
        prompt = _build_prompt(result_df)

        # 3. 發送請求
        response = model.generate_content(prompt)
        
        return response.text

    except Exception as e:
        return f"❌ 生成失敗：{str(e)}\n請確認 API Key 是否正確，或嘗試更換 Google 帳號申請 Key。"

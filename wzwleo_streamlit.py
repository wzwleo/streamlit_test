# -*- coding: utf-8 -*-
"""
Created on Tue May 27 14:48:37 2025

@author: leo20
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

import google.generativeai as genai

genai.configure(api_key="AIzaSyCdX20r2THeDoAXYhRiGJgEHM8gOWNP-ko")

plt.rcParams['font.family'] = ['Microsoft JhengHei']  # 微軟正黑體，Windows系統常用

st.title("Streamlit作業練習")

st.sidebar.title("側邊欄")
    
# 創建多個標籤頁
tab1, tab2, tab3, tab4, tab5= st.tabs(["基礎功能", "Gemini API的連接", "功能二", "功能三", "功能四"])

with tab1:
    st.header("資料上傳")
    st.info("**支援的檔案格式：** CSV、Excel (xlsx / xls)")
    
    # 文件上傳
    uploaded_file = st.file_uploader("拖放文件到這裡", type=['csv'], help="這是傳文件的")
    if uploaded_file is not None:
        st.success("文件上傳成功！")
    
    if uploaded_file:
        st.write("檔案預覽：")
        df = pd.read_csv(uploaded_file)
        st.dataframe(df)
        
        st.header("類別分布")
        gender = st.selectbox("選擇欄位",["--請選擇欄位--"] + list(df.columns))
        if gender != "--請選擇欄位--":
            counts = df[gender].value_counts()
            
            fig, ax = plt.subplots(figsize=(12, 6)) 
            
            counts_sorted = counts.sort_index()
            
            counts_sorted.plot(kind='bar', ax=ax, color='skyblue', alpha=0.8)
            ax.set_title(f"{gender}分類數量", fontsize=16, pad=20)
            ax.set_xlabel(gender, fontsize=12)
            ax.set_ylabel("數量", fontsize=12)
            ax.set_xticklabels(counts_sorted.index, rotation=0, fontsize=11, ha='center')
            plt.tight_layout()
            
            st.pyplot(fig)
        
model = genai.GenerativeModel("gemini-1.5-flash")        
with tab2:
    st.header("Gemini API的連接")
    prompt = st.text_input("請輸入你的問題：")
    if st.button("送出"):
        if prompt.strip() == "":
            st.warning("請輸入內容再送出。")
        else:
            with st.spinner("Gemini 正在思考..."):
                response = model.generate_content(prompt)
                st.success("回答：")
                st.markdown(response.text)

with tab3:
    st.header("趨勢分析")
    
    

with tab4:
    st.header("財務報表")
    
'''
streamlit run "C:/專題報告/wzwleo_streamlit.py"
'''

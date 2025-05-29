# -*- coding: utf-8 -*-
"""
Created on Tue May 27 14:48:37 2025

@author: leo20
"""

import streamlit as st
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# 設定支援中文字體（以 Windows 的「微軟正黑體」為例）
matplotlib.rcParams['font.family'] = 'Microsoft JhengHei'
# 解決負號無法顯示問題
matplotlib.rcParams['axes.unicode_minus'] = False

taba, tabb, tabc, tabd = st.tabs(["學姊的範例", "分頁一", "分頁二", "分頁三"])
with taba:
    
    # 創建多個標籤頁
    tab1, tab2, tab3, tab4 = st.tabs(["基礎功能", "銷售分析", "趨勢分析", "財務報表"])
    
    with tab1:
        st.info("**支援的檔案格式：** CSV、Excel (xlsx / xls)")
        
        # 文件上傳
        uploaded_file = st.file_uploader("拖放文件到這裡", type=['csv'], help="這是傳文件的")
        if uploaded_file is not None:
            st.success("文件上傳成功！")
        
        if uploaded_file:
            st.write("檔案預覽：")
            df = pd.read_csv(uploaded_file)
            st.dataframe(df)
            
            st.write("欄位資訊：")
            field_info = []
            for col in df.columns:
                col_data = df[col]
                field_info.append({
                    '欄位名稱': col,
                    '數據類型': str(col_data.dtype),
                    '非空值數量': col_data.count(),
                    '空值數量': col_data.isnull().sum(),
                })
            
            field_df = pd.DataFrame(field_info)
            st.dataframe(field_df, use_container_width=True)
            
            
    
            
    with tab2:
        st.header("銷售分析")
        
        
    
    with tab3:
        st.header("趨勢分析")
        
        
    
    with tab4:
        st.header("財務報表")
        
        
'''
streamlit run "C:/專題報告/test_2.py"
'''

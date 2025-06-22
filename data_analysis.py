# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 21:21:26 2025

@author: leo20
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

font_path = "./fonts/NotoSansCJKtc-Regular.otf"
my_font = fm.FontProperties(fname=font_path)


def data_analysis_tab():
    st.header("資料分析")
    st.subheader("-類別分布")
    if 'df' in st.session_state:
        df = st.session_state['df']
        plt.rcParams['font.family'] = my_font.get_name()

        selected_col = st.selectbox("選擇欄位", ["--請選擇欄位--"] + list(df.columns), key="group")
        
        
        
        if selected_col != "--請選擇欄位--":
            num_categories = df[selected_col].nunique()
            if num_categories > 20:
                st.write(f"{selected_col} 欄位共有 {num_categories} 個不同的類別")
                st.warning("類別太多，可能會看不清楚！")
            else:
                chart_type = st.radio("選擇圖表類型", ("長條圖", "圓餅圖"), horizontal=True)
                if chart_type == "長條圖": 
                    if st.button("分析"):
                        counts = df[selected_col].value_counts()
                        counts_sorted = counts.sort_index()
                        
                        fig, ax = plt.subplots(figsize=(12, 6))
                        counts_sorted.plot(kind='bar', ax=ax, color='skyblue', alpha=0.8)
                        ax.set_title(f"{selected_col}分類數量", fontsize=16, pad=20, fontproperties=my_font)
                        ax.set_xlabel(selected_col, fontsize=12, fontproperties=my_font)
                        ax.set_ylabel("數量", fontsize=12, fontproperties=my_font)
                        ax.set_xticklabels(counts_sorted.index, rotation=0, fontsize=11, ha='center', fontproperties=my_font)
                        st.pyplot(fig)
                else:  # 圓餅圖
                    counts = df[selected_col].value_counts()
                    counts_sorted = counts.sort_index()
                    if st.button("分析"):
                        fig, ax = plt.subplots(figsize=(8, 8))
                        counts_sorted.plot(kind='pie', ax=ax, autopct='%1.1f%%', startangle=90, textprops={'fontproperties': my_font})
                        ax.set_ylabel("")  # 移除 y 軸標籤
                        ax.set_title(f"{selected_col}分類比例", fontsize=16, pad=20, fontproperties=my_font)
                        st.pyplot(fig)
    else:
        st.warning("請先在 基礎功能 上傳資料")

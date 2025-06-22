# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 21:22:06 2025

@author: leo20
"""

import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.linear_model import LinearRegression

font_path = "./fonts/NotoSansCJKtc-Regular.otf"
my_font = fm.FontProperties(fname=font_path)

def train_and_show_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    st.success(f"回歸模型係數：{model.coef_}")
    st.success(f"截距：{model.intercept_}")
    score = model.score(X, y)
    if score >= 0.8:
        comment = "模型很棒，預測效果很好！"
    elif score >= 0.5:
        comment = "模型普通，準確度不高"
    else:
        comment = "模型很差，準確度很低，幾乎沒用"
    st.success(f"R²分數：{score:.4f} ({comment})")
    return model

def linear_regression_tab():
    st.header("線性回歸分析")
    if 'df' in st.session_state:
        df = st.session_state['df']
        
        if st.checkbox("顯示熱力圖"):
            corr = df.corr(numeric_only=True)
            
            # 建立圖形
            fig, ax = plt.subplots()
            
            # 設定中文字體
            plt.rcParams['font.family'] = my_font.get_name()
            
            # 畫熱力圖
            sns.heatmap(
                corr,
                annot=True,
                cmap="coolwarm",
                ax=ax,
                annot_kws={"fontproperties": my_font},
                cbar_kws={"label": "相關係數"}
            )
            
            # 設定中文的 x/y 標籤字體
            ax.set_xticklabels(ax.get_xticklabels(), fontproperties=my_font, rotation=45, ha='right')
            ax.set_yticklabels(ax.get_yticklabels(), fontproperties=my_font, rotation=0)
            
            cbar = ax.collections[0].colorbar
            cbar.ax.yaxis.label.set_font_properties(my_font)  # 顏色條的標籤字體
            for label in cbar.ax.get_yticklabels():
                label.set_fontproperties(my_font) 
            
            # 顯示圖表到 Streamlit
            st.pyplot(fig)



        
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()

        features = st.multiselect("選擇特徵值", numeric_columns, key="LinearRegression_features")
        target = st.selectbox("選擇目標值", ["--請選擇欄位--"] + numeric_columns, key="LinearRegression_target")

        if features and target != "--請選擇欄位--":
            X = df[features]
            y = df[target]
            if len(features) >= 2:
                st.warning("你選擇了超過兩個特徵欄位，繪圖只能用一個特徵欄位。")
                st.warning("還是要只輸出模型分析結果")
                if st.button("確定"):
                    train_and_show_model(X, y)
            else:
                if st.button("開始分析"):
                    model = train_and_show_model(X, y)
                    y_pred = model.predict(X)

                    plt.figure(figsize=(8, 5))
                    plt.scatter(X, y, color='blue', label='實際資料')
                    plt.plot(X, y_pred, color='red', label='回歸線')
                    plt.xlabel(features[0], fontproperties=my_font)
                    plt.ylabel(target, fontproperties=my_font)
                    plt.title(f'{features[0]} vs {target}', fontproperties=my_font)
                    plt.legend(prop=my_font)
                    plt.grid(True)
                    st.pyplot(plt)
    else:
        st.warning("請先在 基礎功能 上傳資料")

# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 20:30:41 2025

@author: leo20
"""

import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder

def guess_task_type(series):
    if pd.api.types.is_numeric_dtype(series):
        if series.nunique() > 10 and pd.api.types.is_float_dtype(series):
            return "regression", "(回歸)"
        else:
            return "classification", "(分類)"
    else:
        return "classification", "(分類)"

def decision_tree_tab():
    st.header("決策樹演算法")
    if 'df' not in st.session_state:
        st.warning("請先在 基礎功能 上傳資料")
        return

    df_original = st.session_state['df']
    df = df_original.copy()

    target = st.selectbox("選擇預測類別", ["--請選擇欄位--"] + list(df.columns), key="DecisionTree_target")
    if target == "--請選擇欄位--":
        return

    if df[target].nunique() == len(df):
        st.warning("該欄位沒有預測的意義")
        return

    # 移除無意義欄位
    for col in df.columns:
        if df[col].nunique() == len(df):
            df = df.drop(columns=[col])
    df = df.drop(columns=[target])

    st.subheader("數值型資料")
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    inputs_numeric = {}
    for i in range(0, len(numeric_cols), 2):
        cols = st.columns(min(2, len(numeric_cols) - i))
        for j, col_name in enumerate(numeric_cols[i:i+2]):
            inputs_numeric[col_name] = cols[j].number_input(
                f"{col_name}", key=f"num_{col_name}",
                min_value=0, max_value=int(df[col_name].max()), step=1
            )

    st.subheader("非數值型資料")
    non_numeric_cols = df.select_dtypes(exclude=['number']).columns.tolist()
    inputs_non_numeric = {}
    encoding_dict = {}

    for col in non_numeric_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoding_dict[col] = dict(zip(le.classes_, le.transform(le.classes_)))
        options = list(encoding_dict[col].keys())
        selected = st.selectbox(f"{col}", options, key=f"cat_{col}")
        inputs_non_numeric[col] = encoding_dict[col][selected]

    if st.button("開始預測"):
        task_type, explanation = guess_task_type(df_original[target])
        st.write(f"偵測到的任務類型為：{task_type}{explanation}")

        if task_type == "regression":
            model = DecisionTreeRegressor()
        else:
            model = DecisionTreeClassifier()

        x = df
        y = df_original[target]

        model.fit(x, y)

        combined = {**inputs_numeric, **inputs_non_numeric}
        new_data = pd.DataFrame([combined])
        new_data = new_data.reindex(columns=x.columns)

        prediction = model.predict(new_data)
        st.success(f"預測結果: {prediction[0]}")

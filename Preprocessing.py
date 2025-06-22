import streamlit as st
import pandas as pd
import chardet
import io

sample_files = {
    "學生社群媒體成癮": "./Students Social Media Addiction.csv",
    "學生習慣與學業成績": "./student_habits_performance.csv"
}

def clear_data():
    if 'df' in st.session_state:
        del st.session_state['df']

def load_uploaded_file(file):
    if isinstance(file, str):
        with open(file, 'rb') as f:
            raw = f.read(10000)
        result = chardet.detect(raw)
        encoding = result['encoding']
        with open(file, encoding=encoding, errors='replace') as f:
            df = pd.read_csv(f)
    else:
        raw = file.read(10000)
        result = chardet.detect(raw)
        encoding = result['encoding']
        file.seek(0)
        decoded = file.read().decode(encoding, errors='replace')
        df = pd.read_csv(io.StringIO(decoded))
    return df

def upload_data_ui():
    df = None
    status = st.radio(
        "選擇模式：",
        ["上傳檔案", "使用範例資料"],
        horizontal=True
    )

    if status == "使用範例資料":
        selected_file = st.selectbox(
            "選擇範例資料",
            ["--請選擇欄位--"] + list(sample_files.keys()),
            help="以下皆屬公開資料集"
        )
        if selected_file != "--請選擇欄位--":
            df = load_uploaded_file(sample_files[selected_file])
            st.success(f"資料上傳成功！總共有{df.shape[0]}筆資料")
    else:
        uploaded_file = st.file_uploader(
            "拖放資料到這裡", type=['csv'], help="這是傳文件的"
        )
        if uploaded_file is not None:
            df = load_uploaded_file(uploaded_file)
            st.success(f"資料上傳成功！總共有{df.shape[0]}筆資料")

    if df is not None:
        st.write("檔案預覽：")
        st.session_state['df'] = df
        st.dataframe(df)
        
        st.sidebar.dataframe(df)
        
        st.write("欄位資訊：")
        info_df = pd.DataFrame({
            "欄位名稱": df.columns,
            "資料型別": df.dtypes.values,
            "非空值數量": df.count().values,
            "空值數量": df.isnull().sum().values
        })
        st.dataframe(info_df)

    return df

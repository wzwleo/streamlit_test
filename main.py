import streamlit as st
from Preprocessing import upload_data_ui, clear_data
from gemini_api import init_gemini_api, gemini_chat_ui
from data_analysis import data_analysis_tab
from linear_regression_analysis import linear_regression_tab
from Decision_tree import decision_tree_tab


st.set_page_config(page_title="Streamlit作業練習")

def main():
    st.title("Streamlit作業練習")
    st.sidebar.title("側邊欄")

    # 初始化 Gemini API
    init_gemini_api()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "基礎功能🐣", "Gemini API的連接🚀", "資料分析📊", "線性回歸分析📈", "決策樹演算法🌳"
    ])

    with tab1:
        df = upload_data_ui()
        if df is None:
            clear_data()

    with tab2:
        gemini_chat_ui()

    with tab3:
        data_analysis_tab()

    with tab4:
        linear_regression_tab()

    with tab5:
        decision_tree_tab()
    
    

if __name__ == "__main__":
    main()


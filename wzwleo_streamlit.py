import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import chardet
import io
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
# 從環境變數中取得 API 金鑰
api_key = os.getenv("GOOGLE_API_KEY")

# 設定 API 金鑰
genai.configure(api_key=api_key)

# 指定字體檔案路徑（如果放在 fonts 資料夾，就改成 "fonts/MicrosoftJhengHei.ttf"）
font_path = "./fonts/NotoSansCJKtc-Regular.otf"

# 建立字體物件
my_font = fm.FontProperties(fname=font_path)

sample_files = {
    "學生社群媒體成癮": "Students Social Media Addiction.csv",
    "學生習慣與學業成績":"student_habits_performance.csv"
}

def clear_data():
    if 'df' in st.session_state:
        del st.session_state['df']

def load_sample_data(path):
    return pd.read_csv(path)

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

def train_and_show_model(X, y, hobbies, gender1):
    model = LinearRegression()
    model.fit(X, y)
    st.success(f"回歸模型係數：{model.coef_}")
    st.success(f"截距：{model.intercept_}")
    ms = model.score(X, y)
    if ms >= 0.8:
        ms_comment = "模型很棒，預測效果很好！"
    elif ms >= 0.5:
        ms_comment = "模型普通，準確度不高"
    else:
        ms_comment = "模型很差，準確度很低，幾乎沒用"
    st.success(f"R²分數：{model.score(X, y):.4f} ({ms_comment})")
    return model

st.title("Streamlit作業練習")

st.sidebar.title("側邊欄")  
 
# 創建多個標籤頁
tab1, tab2, tab3, tab4, tab5= st.tabs(["基礎功能🐣", "Gemini API的連接🚀", "資料分析📊", "線性回歸分析📈", "決策樹演算法🌳"])

with tab1:
    df = None
    st.header("資料上傳")
    status = st.radio(
        "選擇模式：",
        ["上傳檔案", "使用範例資料"],
        horizontal=True
    )
    if status == "使用範例資料":
        uploaded_file = st.selectbox("選擇範例資料",["--請選擇欄位--"] + list(sample_files.keys()), help="以下皆屬公開資料集")
        if uploaded_file != "--請選擇欄位--":
            file_path = sample_files[uploaded_file]
            df = load_uploaded_file(file_path)
            st.success(f"資料上傳成功！總共有{df.shape[0]}筆資料")
        elif uploaded_file == "--請選擇欄位--":
            clear_data()
            
            
    elif status == "上傳檔案":
        st.info("**支援的檔案格式：** CSV、Excel (xlsx / xls)")
        
        # 文件上傳
        uploaded_file = st.file_uploader("拖放資料到這裡", type=['csv'], help="這是傳文件的")
        if uploaded_file is not None:
            df = load_uploaded_file(uploaded_file)     
            st.success(f"資料上傳成功！總共有{df.shape[0]}筆資料")
            
        else:
            clear_data()
            
    if df is not None:
        st.write("檔案預覽：")
        st.session_state['df'] = df
        st.dataframe(df)
        
        st.write("欄位資訊：")      
        info_df = pd.DataFrame({
            "欄位名稱": df.columns,
            "資料型別": df.dtypes.values,
            "非空值數量": df.count().values,
            "空值數量": df.isnull().sum().values
        })
    
        st.dataframe(info_df)        

if 'df' in st.session_state:
    df = st.session_state['df']
    st.sidebar.write("檔案預覽")
    st.sidebar.dataframe(df)
else:
    st.write("")
        
        
if "chat" not in st.session_state:
    model = genai.GenerativeModel("gemini-1.5-flash")
    st.session_state.chat = model.start_chat(history=[])

chat = st.session_state.chat  # 取出 chat 對象
with tab2:
    st.header("Gemini API的連接")
    prompt = st.text_input("請輸入你的問題：")
    if st.button("送出"):
        if prompt.strip() == "":
            st.warning("請輸入內容再送出。")
        else:
            with st.spinner("Gemini 正在思考..."):
                response = chat.send_message(prompt)
                st.success("回答：")
                st.markdown(response.text)
                
                st.divider()
                st.subheader("對話紀錄：")
                for message in chat.history:
                    role = "使用者" if message.role == "user" else "Gemini"
                    content = message.parts[0].text
                    st.markdown(f"**{role}**: {content}")

with tab3:
    st.header("資料分析")
    st.subheader("-類別分布")
    if 'df' in st.session_state:
        df = st.session_state['df']
        plt.rcParams['font.family'] = my_font.get_name()
        
        gender = st.selectbox("選擇欄位",["--請選擇欄位--"] + list(df.columns), key="group")
        
        if gender != "--請選擇欄位--":
            num_categories = df[gender].nunique()
            
            # 如果類別太多（例如超過20），可以顯示警告
            if num_categories > 20:
                st.write(f"{gender} 欄位共有 {num_categories} 個不同的類別")
                st.warning("類別太多，可能會看不清楚！")
            else:    
                counts = df[gender].value_counts()
                counts_sorted = counts.sort_index()
                fig, ax = plt.subplots(figsize=(12, 6)) 
                
                
                counts_sorted.plot(kind='bar', ax=ax, color='skyblue', alpha=0.8)
                ax.set_title(f"{gender}分類數量", fontsize=16, pad=20, fontproperties=my_font)
                ax.set_xlabel(gender, fontsize=12, fontproperties=my_font)
                ax.set_ylabel("數量", fontsize=12, fontproperties=my_font)
                ax.set_xticklabels(counts_sorted.index, rotation=0, fontsize=11, ha='center', fontproperties=my_font)

                
                st.pyplot(fig)
    else:
        st.warning("請先在 基礎功能 上傳資料")
    
    

with tab4:
    st.header("線性回歸分析")
    
    if 'df' in st.session_state:
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()

        hobbies = st.multiselect("選擇特徵值", numeric_columns, key="LinearRegression_1")
        gender = st.selectbox("選擇目標值", ["--請選擇欄位--"] + numeric_columns, key="LinearRegression_2")
        if hobbies and gender != "--請選擇欄位--":
           X = df[hobbies]
           y = df[gender]  
           if len(hobbies) >= 2:
              st.warning("你選擇了超過兩個特徵欄位，繪圖只能用一個特徵欄位。")
              st.warning("還是要只輸出模型分析結果")
              if st.button("確定"):
                  model = train_and_show_model(X, y, hobbies, gender)
           else:
                if st.button("開始分析"):                      
                    model = train_and_show_model(X, y, hobbies, gender)
                    y_pred = model.predict(X)
            
                    plt.figure(figsize=(8, 5))
                    
                    plt.scatter(X, y, color='blue', label='實際資料')
                    plt.plot(X, y_pred, color='red', label='回歸線')
                    
                    plt.xlabel(hobbies[0], fontproperties=my_font)
                    plt.ylabel(f'{gender}', fontproperties=my_font)
                    plt.title(f'{hobbies[0]} vs {gender}', fontproperties=my_font)
                    plt.legend(prop=my_font)
                    plt.grid(True)
                    
                    st.pyplot(plt)
        
    else:
        st.warning("請先在 基礎功能 上傳資料")

with tab5:
    st.header("決策樹演算法")
    df_2 = df
    if 'df' in st.session_state:
       df = st.session_state['df']       
       gender = st.selectbox("選擇預測類別",["--請選擇欄位--"] + list(df.columns), key="DecisionTree")
       if gender != "--請選擇欄位--":    
           unique_count = df[gender].nunique()
           total_rows = len(df)
           if unique_count == total_rows:
               st.warning("該欄位沒有預測的意義")
           else:
               for i in df.columns:
                   unique_count = df[i].nunique()
                   
                   total_rows = len(df)
                   if unique_count == total_rows:
                      df = df.drop(columns=[i])     
               df = df.drop(columns=[gender])  

               st.subheader("數值型資料")
               numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
                
               inputs_numeri = {}
               max_cols_per_row = 2
               for i in range(0, len(numeric_cols), max_cols_per_row):
                   cols = st.columns(min(max_cols_per_row, len(numeric_cols) - i))
                   for j, col_name in enumerate(numeric_cols[i:i+max_cols_per_row]):
                       inputs_numeri[col_name] = cols[j].number_input(f"{col_name}", key=col_name, max_value=int(df[col_name].max()), min_value=0, step=1)
               
               
               st.subheader("非數值型資料")
               # 取得非數值欄位清單
               non_numeric_cols = df.select_dtypes(exclude=['number']).columns.tolist()
               
               encoding_dict = {}
               
               for col in non_numeric_cols:
                   le = LabelEncoder()
                   df[col] = le.fit_transform(df[col].astype(str))
                   # 把每個標籤對應的原始字串存在字典裡
                   encoding_dict[col] = dict(zip(le.classes_, le.transform(le.classes_)))
                   # 將字串形式的數字轉成整數
                   encoding_dict[col] = {k: int(v) for k, v in encoding_dict[col].items()}
                
               inputs_non_numeri = {}
               for i in range(0, len(non_numeric_cols), max_cols_per_row):
                   cols = st.columns(min(max_cols_per_row, len(non_numeric_cols) - i))
                   for j, col_name in enumerate(non_numeric_cols[i:i+max_cols_per_row]):
                       inputs_non_numeri[col_name] = cols[j].number_input(f"{col_name}", key=col_name, min_value=0, max_value=df[col_name].nunique()-1, step=1)
               
               
               
               st.subheader("非數值型資料 數字對應表")
               st.write(encoding_dict)
               
               if st.button("開始預測"):
                   combined = {**inputs_numeri, **inputs_non_numeri}
                
                   x = df
                   y = df_2[[gender]]
                   
                   model = DecisionTreeClassifier()
                   model.fit(x, y)
                   
                   new_data = pd.DataFrame([combined])
                   
                   new_data = new_data.reindex(columns=x.columns)
                   prediction = model.predict(new_data)
                   predicted_value = prediction[0]
                   
                   st.success(f"預測結果: {predicted_value}")
                   df.to_csv("output.csv", index=False, encoding="utf-8-sig")
 
    else:
        st.warning("請先在 基礎功能 上傳資料")
        
        
        
   


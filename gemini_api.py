# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 21:17:59 2025

@author: leo20
"""

import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

def init_gemini_api():
    
    if "chat" not in st.session_state:
        model = genai.GenerativeModel("gemini-1.5-flash")
        st.session_state.chat = model.start_chat(history=[])

def gemini_chat_ui():
    chat = st.session_state.chat
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

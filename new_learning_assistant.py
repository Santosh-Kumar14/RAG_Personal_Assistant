import streamlit as st
from learning_assistant_functions import *
import os

def initialize_session_state():
    if 'questions' not in st.session_state:
        st.session_state.questions = []
    if 'llm' not in st.session_state:
        st.session_state.llm = initialize_llm("api")
    if 'qa_chain' not in st.session_state:
        st.session_state.qa_chain = None
    if 'file_uploaded' not in st.session_state:
        st.session_state.file_uploaded = False

initialize_session_state()

st.set_page_config(layout="centered")

st.markdown("""
<style>
.stApp {
    background: linear-gradient(to bottom right, #2c1f4a, #1a103a);
}
.big-font {
    font-size: 40px !important;
    font-weight: bold;
    color: white;
    margin-bottom: 40px;
    text-align: center;
}
.answer-box {
    background-color: rgba(255, 255, 255, 0.05);
    border-radius: 10px;
    padding: 20px;
    margin-top: 20px;
    color: white;
}
.stButton>button {
    background-color: #a29bfe;
    color: black;
    font-weight: bold;
    border-radius: 20px;
    padding: 10px 20px;
    font-size: 16px;
    border: none;
}
.spacer {
    margin-top: 30px;
}
.upload-box {
    border: 2px dashed #6c5ce7;
    border-radius: 10px;
    padding: 20px;
    text-align: center;
    background-color: rgba(108, 92, 231, 0.1);
    margin-bottom: 20px;
}
.upload-text {
    color: #a29bfe;
    font-size: 16px;
}
.stFileUploader {
    display: none;
}
.stCheckbox label {
    color: white !important;
}
.stCheckbox label span {
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-font">Learning Assistant</p>', unsafe_allow_html=True)

if not st.session_state.file_uploaded:
    with st.container():
        uploaded_file = st.file_uploader("", type=["csv", "txt", "pdf"])
        st.markdown('<p class="upload-text">Click to upload or drag and drop</p>', unsafe_allow_html=True)
    
    if st.button('Upload and Generate Questions'):
        if uploaded_file is not None:
            st.write("Processing your file...")
            st.session_state.questions = generate_questions(st.session_state.llm, uploaded_file.getvalue(), 3)
            st.session_state.qa_chain = create_retrieval_qa_chain(st.session_state.llm, uploaded_file.getvalue())
            st.session_state.file_uploaded = True
            st.rerun()
        else:
            st.write("Please upload a file first.")
else:
    selected_questions = []
    for question in st.session_state.questions:
        if st.checkbox(question):
            selected_questions.append(question)

    if st.button('Answer Selected Questions'):
        if selected_questions:
            answers_html = ""
            for i, question in enumerate(selected_questions):
                answer = st.session_state.qa_chain({"query": question})["result"]
                answers_html += f"<strong>Q: {question}</strong><br><br><strong>A:</strong> {answer}"
                if i < len(selected_questions) - 1:
                    answers_html += "<br><br>---<br><br>"
                else:
                    answers_html += "<br><br>"  
            st.markdown(f'<div class="answer-box">{answers_html}</div>', unsafe_allow_html=True)
        else:
            st.write("Please select at least one question to answer.")

    if st.button('Upload New File'):
        st.session_state.file_uploaded = False
        st.session_state.questions = []
        st.session_state.qa_chain = None
        st.rerun()
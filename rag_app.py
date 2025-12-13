import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
import os

# ------------ ENV & OPENAI SETUP ------------
load_dotenv()  # loads OPENAI_API_KEY from .env or Streamlit secrets

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# ------------ PAGE CONFIG + HIDE MENU ------------
st.set_page_config(page_title="DocuChat AI", page_icon=":brain:", layout="wide")

hide_st_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)

# ------------ HEADER ------------
st.title("🧠 DocuChat AI")
st.caption("🚀 Powered by OpenAI | Chat with Multiple PDFs")

# ------------ SIDEBAR (UPLOAD + PROCESS) ------------
with st.sidebar:
    st.header("📂 Document Center")

    pdf_docs = st.file_uploader(
        "Upload your PDFs",
        type="pdf",
        accept_multiple_files=True
    )

    if st.button("Processing Documents"):
        if not OPENAI_API_KEY:
            st.error("OPENAI_API_KEY is missing. Please add it in Streamlit secrets.")
        elif pdf_docs:
            with st.spinner("Processing... This may take a moment."):
                # A. Extract text from all PDFs
                raw_text = ""
                for pdf in pdf_docs:
                    reader = PdfReader(pdf)
                    for page in reader.pages:
                        text = page.extract_text()
                        if text:
                            raw_text += text

                # B. Split text into chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len,
                )
                chunks = text_splitter.split_text(raw_text)

                # C. Create vector store (needs OPENAI_API_KEY)
                embeddings = OpenAIEmbeddings(
                    model="text-embedding-3-small",
                    openai_api_key=OPENAI_API_KEY,
                )  # [web:192][web:223]
                vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)

                # D. Save to session state
                st.session_state.vectorstore = vectorstore
                st.session_state.messages = []
                st.success("✅ Documents Processed!")
        else:
            st.warning("⚠️ Please upload at least one PDF first.")

    st.divider()

    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.experimental_rerun()

# ------------ SESSION STATE INIT ------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# ------------ SHOW CHAT HISTORY ------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ------------ CHAT INPUT ------------
user_question = st.chat_input("Ask a question about your documents...")

if user_question:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.write(user_question)

    if st.session_state.vectorstore is None:
        with st.chat_message("assistant"):
            st.warning("⚠️ Please upload and process your PDFs first!")
    else:
        llm = ChatOpenAI(
            model="gpt-4.1-mini",        # use any chat model you enabled
            api_key=OPENAI_API_KEY,
        )  # [web:229]
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=st.session_state.vectorstore.as_retriever(),
            return_source_documents=False,
        )

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = qa_chain({"question": user_question, "chat_history": []})
                answer = result["answer"]
                st.write(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})

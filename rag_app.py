import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain


def get_openai_api_key() -> str:
    # Works locally with .env and on Streamlit Cloud with secrets.
    load_dotenv()
    return st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))


def extract_text_from_pdfs(pdf_files) -> str:
    raw_text = ""
    for pdf in pdf_files:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                raw_text += text + "\n"
    return raw_text


def build_vectorstore(raw_text: str) -> FAISS:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = splitter.split_text(raw_text)

    if not chunks:
        raise ValueError(
            "No extractable text found in the uploaded PDFs. "
            "If these are scanned PDFs (images), run OCR first."
        )

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=get_openai_api_key(),
    )
    return FAISS.from_texts(texts=chunks, embedding=embeddings)


# ------------ PAGE CONFIG + HIDE MENU ------------
st.set_page_config(page_title="DocuChat AI", page_icon="🧠", layout="wide")

st.markdown(
    """
    <style>
      #MainMenu {visibility: hidden;}
      footer {visibility: hidden;}
      header {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------ SESSION STATE INIT ------------
st.session_state.setdefault("messages", [])
st.session_state.setdefault("vectorstore", None)

# ------------ TOP AREA: TITLE + UPLOAD ------------
left_col, right_col = st.columns([2, 1])

with left_col:
    st.title("🧠 DocuChat AI")
    st.caption("Powered by OpenAI | Chat with Multiple PDFs")

with right_col:
    st.subheader("📂 Document Center")

    pdf_docs = st.file_uploader(
        "Upload your PDFs",
        type="pdf",
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    col_a, col_b = st.columns(2)

    with col_a:
        process_clicked = st.button("Process Documents", use_container_width=True)

    with col_b:
        clear_clicked = st.button("Clear Chat", use_container_width=True)

    if clear_clicked:
        st.session_state.messages = []
        st.session_state.vectorstore = None
        st.rerun()

    if process_clicked:
        api_key = get_openai_api_key()
        if not api_key:
            st.error("OPENAI_API_KEY is missing. Add it to Streamlit secrets or your environment.")
        elif not pdf_docs:
            st.warning("Please upload at least one PDF first.")
        else:
            try:
                with st.spinner("Processing... This may take a moment."):
                    raw_text = extract_text_from_pdfs(pdf_docs)
                    if not raw_text.strip():
                        st.error(
                            "Could not extract any text from the PDFs. "
                            "If they are scanned/image PDFs, you need OCR."
                        )
                    else:
                        st.session_state.vectorstore = build_vectorstore(raw_text)
                        st.session_state.messages = []
                        st.success("✅ Documents processed!")
            except Exception as e:
                st.exception(e)

st.markdown("---")

# ------------ SHOW CHAT HISTORY ------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ------------ CHAT INPUT / QA ------------
user_question = st.chat_input("Ask a question about your documents...")

if user_question:
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.write(user_question)

    if st.session_state.vectorstore is None:
        with st.chat_message("assistant"):
            st.warning("Please upload and process your PDFs first.")
    else:
        api_key = get_openai_api_key()
        if not api_key:
            with st.chat_message("assistant"):
                st.error("OPENAI_API_KEY is missing.")
        else:
            # IMPORTANT: use openai_api_key (not api_key) for common langchain_openai versions
            llm = ChatOpenAI(
                model="gpt-4.1-mini",
                openai_api_key=api_key,
                temperature=0,
            )

            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4}),
                return_source_documents=False,
            )

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Use stored history (as tuples) instead of always passing []
                    history = [
                        (m["content"], st.session_state.messages[i + 1]["content"])
                        for i, m in enumerate(st.session_state.messages[:-1])
                        if m["role"] == "user" and i + 1 < len(st.session_state.messages)
                        and st.session_state.messages[i + 1]["role"] == "assistant"
                    ]
                    result = qa_chain({"question": user_question, "chat_history": history})
                    answer = result.get("answer", "")
                    st.write(answer)

            st.session_state.messages.append({"role": "assistant", "content": answer})

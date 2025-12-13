import os
import streamlit as st

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

# -------------------- Page Config --------------------
st.set_page_config(page_title="DocuChat AI", page_icon="📄", layout="wide")

# -------------------- Secrets / Keys --------------------
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))

# -------------------- Helpers --------------------
def extract_text_from_pdfs(pdf_files) -> str:
    text_parts = []
    for f in pdf_files:
        reader = PdfReader(f)
        for page in reader.pages:
            text_parts.append(page.extract_text() or "")
    return "\n".join(text_parts)


def build_vectorstore(raw_text: str) -> FAISS:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = splitter.split_text(raw_text)

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=OPENAI_API_KEY,
    )
    return FAISS.from_texts(chunks, embedding=embeddings)


def build_chain(vectorstore: FAISS):
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=OPENAI_API_KEY,
        temperature=0.2,
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=False,
    )
    return chain


# -------------------- UI --------------------
st.title("DocuChat AI")
st.write("Upload PDFs, build a local FAISS index, and chat with your documents.")

with st.sidebar:
    st.header("1) Upload PDFs")
    pdf_docs = st.file_uploader(
        "Upload one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )

    st.header("2) Build knowledge base")
    build_btn = st.button("Process Documents", type="primary")

    st.divider()
    st.caption("Set the API key in Streamlit Cloud → App → Settings → Secrets:")
    st.code('OPENAI_API_KEY="..."', language="toml")

# Session state init
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chain" not in st.session_state:
    st.session_state.chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# Validate key early
if not OPENAI_API_KEY:
    st.warning("OPENAI_API_KEY not found. Add it in Streamlit Secrets to use this app.")

# Build vectorstore + chain
if build_btn:
    if not pdf_docs:
        st.warning("Please upload at least one PDF first.")
    elif not OPENAI_API_KEY:
        st.error("Missing OPENAI_API_KEY. Add it to Streamlit Secrets.")
    else:
        with st.spinner("Processing PDFs and building vector index..."):
            raw_text = extract_text_from_pdfs(pdf_docs)
            st.session_state.vectorstore = build_vectorstore(raw_text)
            st.session_state.chain = build_chain(st.session_state.vectorstore)
            st.session_state.messages = []
        st.success("Documents processed! You can now chat.")

# Chat UI
st.subheader("Chat")

# Render previous chat
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Ask something about your PDFs...")
if prompt:
    if not st.session_state.chain:
        st.info("Process your documents first (left sidebar).")
    else:
        # show user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Convert message list into (human, ai) tuples expected by ConversationalRetrievalChain
        history = []
        last_user = None
        for m in st.session_state.messages:
            if m["role"] == "user":
                last_user = m["content"]
            elif m["role"] == "assistant" and last_user is not None:
                history.append((last_user, m["content"]))
                last_user = None

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = st.session_state.chain(
                    {"question": prompt, "chat_history": history}
                )
                answer = result["answer"]
                st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})

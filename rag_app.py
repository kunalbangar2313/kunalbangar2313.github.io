import os
import streamlit as st

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate


def build_vectorstore(raw_text: str, api_key: str):
    """
    Build a FAISS vectorstore from raw text using OpenAI embeddings.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""],
    )
    docs = splitter.create_documents([raw_text])

    # NOTE: This uses the modern openai-python client under the hood.
    # Do NOT pass proxies here; keep deps aligned in requirements.txt.
    embeddings = OpenAIEmbeddings(
        api_key=api_key,
        model="text-embedding-3-small",
    )

    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore


def answer_question(vectorstore, question: str, api_key: str) -> str:
    """
    Retrieve relevant chunks and answer with a chat model.
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    docs = retriever.get_relevant_documents(question)

    context = "\n\n---\n\n".join(d.page_content for d in docs)

    llm = ChatOpenAI(
        api_key=api_key,
        model="gpt-4o-mini",
        temperature=0.2,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Answer using ONLY the provided context. "
                "If the answer is not in the context, say: 'I don't know based on the provided text.'",
            ),
            ("user", "Context:\n{context}\n\nQuestion:\n{question}"),
        ]
    )

    return (prompt | llm).invoke({"context": context, "question": question}).content


def main():
    st.set_page_config(page_title="RAG App", layout="wide")
    st.title("RAG App (FAISS + OpenAI Embeddings)")

    # API key: prefer input, fallback to env var
    api_key = st.text_input("OpenAI API Key", type="password")
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY", "")

    if not api_key:
        st.info("Enter your OpenAI API key to continue (or set OPENAI_API_KEY).")
        st.stop()

    st.subheader("Build Knowledge Base")
    raw_text = st.text_area("Paste your text here", height=260, placeholder="Paste your documents / notes here...")

    col1, col2 = st.columns([1, 2])

    with col1:
        if st.button("Build / Rebuild Vectorstore", type="primary", disabled=not raw_text.strip()):
            with st.spinner("Building vectorstore..."):
                st.session_state.vectorstore = build_vectorstore(raw_text, api_key)
            st.success("Vectorstore built and saved in session.")

    with col2:
        if "vectorstore" in st.session_state:
            st.success("Vectorstore is ready.")
        else:
            st.warning("Vectorstore not built yet.")

    st.divider()

    st.subheader("Ask Questions")
    question = st.text_input("Your question", placeholder="Ask something about the pasted text...")

    if st.button("Answer", disabled=("vectorstore" not in st.session_state) or (not question.strip())):
        with st.spinner("Retrieving & generating answer..."):
            response = answer_question(st.session_state.vectorstore, question, api_key)
        st.markdown("### Answer")
        st.write(response)


if __name__ == "__main__":
    main()

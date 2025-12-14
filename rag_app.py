import os
import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate


APP_TITLE = "RAG App (FAISS + OpenAI)"


def get_api_key() -> str:
    """Get OpenAI API key from Streamlit secrets, env var, or user input."""
    key_from_secrets = ""
    try:
        key_from_secrets = st.secrets.get("OPENAI_API_KEY", "")
    except Exception:
        key_from_secrets = ""

    key_from_env = os.getenv("OPENAI_API_KEY", "")

    key_input = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-...",
    )

    return (key_input or key_from_secrets or key_from_env or "").strip()


def read_uploaded_files(files: list) -> list[Document]:
    """Read .txt, .md, .pdf files into LangChain Document objects."""
    docs: list[Document] = []

    for f in files:
        name = f.name
        suffix = name.lower().split(".")[-1] if "." in name else ""

        if suffix in ("txt", "md"):
            text = f.read().decode("utf-8", errors="ignore")
            if text.strip():
                docs.append(Document(page_content=text, metadata={"source": name}))

        elif suffix == "pdf":
            from pypdf import PdfReader

            reader = PdfReader(f)
            pages = []
            for page in reader.pages:
                page_text = page.extract_text() or ""
                if page_text.strip():
                    pages.append(page_text)

            text = "\n\n".join(pages)
            if text.strip():
                docs.append(Document(page_content=text, metadata={"source": name}))

        else:
            st.warning(
                f"Skipped unsupported file: {name} (supported: .txt, .md, .pdf)"
            )

    return docs


def build_vectorstore(docs: list[Document], api_key: str) -> FAISS:
    """Create a FAISS vectorstore from documents using OpenAI embeddings."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=api_key,
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


def answer_question(vectorstore: FAISS, question: str, api_key: str) -> str:
    """Retrieve relevant chunks and answer the question with GPT."""
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
                "If the answer is not in the context, say: "
                "'I don't know based on the provided documents.'",
            ),
            ("user", "Context:\n{context}\n\nQuestion:\n{question}"),
        ]
    )

    chain = prompt | llm
    result = chain.invoke({"context": context, "question": question})
    return result.content


def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    api_key = get_api_key()
    if not api_key:
        st.info("Add your OpenAI API key (secrets/env/input) to continue.")
        st.stop()

    st.subheader("1) Upload documents")
    files = st.file_uploader(
        "Upload .txt, .md, or .pdf files",
        type=["txt", "md", "pdf"],
        accept_multiple_files=True,
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        if st.button(
            "Build / Rebuild Vectorstore",
            type="primary",
            disabled=not files,
        ):
            with st.spinner("Reading files..."):
                docs = read_uploaded_files(files)

            if not docs:
                st.error("No text could be extracted from the uploaded files.")
                st.stop()

            with st.spinner("Building vectorstore (embeddings + FAISS)..."):
                try:
                    st.session_state.vectorstore = build_vectorstore(docs, api_key)
                except Exception as e:
                    st.exception(e)
                    st.stop()

            st.success("Vectorstore built.")

    with col2:
        if "vectorstore" in st.session_state:
            st.success("Vectorstore is ready.")
        else:
            st.warning("Vectorstore not built yet.")

    st.divider()

    st.subheader("2) Ask questions")
    question = st.text_input(
        "Question",
        placeholder="Ask something about your uploaded documents...",
    )

    btn_disabled = ("vectorstore" not in st.session_state) or (not question.strip())

    if st.button("Answer", disabled=btn_disabled):
        with st.spinner("Retrieving & generating answer..."):
            try:
                response = answer_question(
                    st.session_state.vectorstore,
                    question,
                    api_key,
                )
            except Exception as e:
                st.exception(e)
                st.stop()

        st.markdown("### Answer")
        st.write(response)


if __name__ == "__main__":
    main()

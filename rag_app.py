import os
import hashlib
import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate


APP_TITLE = "RAG App (FAISS + OpenAI)"


def get_api_key() -> str:
    # Prefer Streamlit secrets, then env var, then UI input.
    key_from_secrets = ""
    try:
        key_from_secrets = st.secrets.get("OPENAI_API_KEY", "")
    except Exception:
        key_from_secrets = ""

    key_from_env = os.getenv("OPENAI_API_KEY", "")

    key_input = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")

    return (key_input or key_from_secrets or key_from_env or "").strip()


def read_uploaded_files(files: list) -> list[Document]:
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
            st.warning(f"Skipped unsupported file: {name} (supported: .txt, .md, .pdf)")

    return docs


def compute_docs_fingerprint(docs: list[Document]) -> str:
    """Compute a stable hash of documents based on content and metadata."""
    content_parts = []
    for doc in sorted(docs, key=lambda d: d.metadata.get("source", "")):
        content_parts.append(doc.page_content)
        content_parts.append(str(sorted(doc.metadata.items())))
    
    combined = "\n".join(content_parts)
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


def get_embeddings(api_key: str = ""):
    """Get embeddings model based on environment configuration."""
    use_local = os.getenv("USE_LOCAL_EMBEDDINGS", "").strip().lower() in ("1", "true", "yes")
    
    if use_local:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    else:
        return OpenAIEmbeddings(
            api_key=api_key,
            model="text-embedding-3-small",
        )


def build_vectorstore(docs: list[Document], api_key: str) -> FAISS:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    # Compute fingerprint for caching
    fingerprint = compute_docs_fingerprint(docs)
    cache_dir = os.path.join(".vectorstore", fingerprint)
    
    # Try to load from cache
    if os.path.exists(cache_dir):
        try:
            embeddings = get_embeddings(api_key)
            vectorstore = FAISS.load_local(
                cache_dir,
                embeddings,
                allow_dangerous_deserialization=True
            )
            st.info(f"✓ Loaded vectorstore from cache: {cache_dir}")
            return vectorstore
        except Exception as e:
            st.warning(f"Cache load failed, rebuilding: {e}")
    
    # Build fresh vectorstore
    embeddings = get_embeddings(api_key)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # Save to cache
    try:
        os.makedirs(cache_dir, exist_ok=True)
        vectorstore.save_local(cache_dir)
        st.info(f"✓ Saved vectorstore to cache: {cache_dir}")
    except Exception as e:
        st.warning(f"Failed to save cache (will rebuild next time): {e}")
    
    return vectorstore


def answer_question(vectorstore: FAISS, question: str, api_key: str) -> str:
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
                "If the answer is not in the context, say: 'I don't know based on the provided documents.'",
            ),
            ("user", "Context:\n{context}\n\nQuestion:\n{question}"),
        ]
    )

    return (prompt | llm).invoke({"context": context, "question": question}).content



def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    use_local = os.getenv("USE_LOCAL_EMBEDDINGS", "").strip().lower() in ("1", "true", "yes")
    
    api_key = get_api_key()
    if not api_key and not use_local:
        st.info("Add your OpenAI API key (Streamlit secrets/env/UI) to continue, or set USE_LOCAL_EMBEDDINGS=1 to use local embeddings.")
        st.stop()
    
    if use_local:
        st.info("🏠 Using local embeddings (HuggingFace sentence-transformers)")

    st.subheader("1) Upload documents")
    files = st.file_uploader(
        "Upload .txt, .md, or .pdf files",
        type=["txt", "md", "pdf"],
        accept_multiple_files=True,
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        if st.button("Build / Rebuild Vectorstore", type="primary", disabled=not files):
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
    question = st.text_input("Question", placeholder="Ask something about your uploaded documents...")

    if st.button("Answer", disabled=("vectorstore" not in st.session_state) or (not question.strip())):
        with st.spinner("Retrieving & generating answer..."):
            try:
                response = answer_question(st.session_state.vectorstore, question, api_key)
            except Exception as e:
                st.exception(e)
                st.stop()

        st.markdown("### Answer")
        st.write(response)


if __name__ == "__main__":
    main()

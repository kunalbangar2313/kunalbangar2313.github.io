import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Page Config ---
st.set_page_config# --- Hide Streamlit Menu and Footer ---
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.set_page_config(page_title="DocuChat AI", page_icon=":brain:", layout="wide")


# --- Custom CSS for UI Polish ---
st.markdown("""
<style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .stButton button {
        width: 100%;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.title("🧠 DocuChat AI")
st.caption("🚀 Powered by OpenAI | Chat with Multiple PDFs")

# --- Sidebar ---
with st.sidebar:
    st.header("📂 Document Center")
    
    # 1. FILE UPLOADER (Accepts Multiple Files)
    pdf_docs = st.file_uploader(
        "Upload your PDFs", 
        accept_multiple_files=True, 
        type="pdf"
    )
    
    # Process Button
    if st.button("Processing Documents"):
        if pdf_docs:
            with st.spinner("Processing... This may take a moment."):
                # A. Extract Text from ALL PDFs
                raw_text = ""
                for pdf in pdf_docs:
                    pdf_reader = PdfReader(pdf)
                    for page in pdf_reader.pages:
                        raw_text += page.extract_text()
                
                # B. Split Text into Chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len
                )
                chunks = text_splitter.split_text(raw_text)
                
                # C. Create Vector Store (The "Brain")
                embeddings = OpenAIEmbeddings()
                vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
                
                # D. Save to Session State
                st.session_state.vectorstore = vectorstore
                st.session_state.chat_history = []  # Reset chat history on new upload
                
                st.success("✅ Documents Processed!")
        else:
            st.warning("⚠️ Please upload a PDF first.")

    st.divider()
    
    # 2. CLEAR CHAT BUTTON
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

# --- Chat Logic ---

# Initialize Chat History in Session State if not present
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# Display Chat History (So it remembers previous messages)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User Input
user_question = st.chat_input("Ask a question about your documents...")

if user_question:
    # 1. Display User Message
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.write(user_question)

    # 2. Check if we have processed documents
    if st.session_state.vectorstore is not None:
        # Create the Conversation Chain
        llm = ChatOpenAI()
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=st.session_state.vectorstore.as_retriever(),
            return_source_documents=True
        )
        
        # Get Response (passing simple chat history list)
        # We handle history manually for Streamlit display, but pass a simple list to LangChain if needed
        # For simplicity here, we just ask the question with context
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Run the chain
                # Note: We pass an empty list for chat_history here to keep it stateless per query 
                # OR we can manage a list of (query, answer) tuples for true context.
                # Let's do simple context:
                history_tuples = [] # You can expand this to use st.session_state.chat_history if you want deep memory
                
                response = qa_chain({"question": user_question, "chat_history": history_tuples})
                answer = response['answer']
                
                st.write(answer)
                
        # Save Assistant Message
        st.session_state.messages.append({"role": "assistant", "content": answer})
    
    else:
        # If no documents uploaded yet
        with st.chat_message("assistant"):
            st.warning("⚠️ Please upload and process a PDF document first!")



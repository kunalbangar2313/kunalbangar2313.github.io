import streamlit as st
import google.generativeai as genai
from ragie import Ragie
import time

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="DocuChat AI", page_icon="🧠", layout="wide")

# --- 2. CUSTOM STYLING ---
st.markdown("""
<style>
    .stChatInput {
        position: fixed;
        bottom: 20px;
        z-index: 1000;
        width: 70%; 
        left: 50%;
        transform: translateX(-50%);
    }
    .main {
        padding-bottom: 100px;
    }
    h1 {
        background: -webkit-linear-gradient(45deg, #4776E6, #8E54E9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. SIDEBAR ---
with st.sidebar:
    st.title("📂 Document Center")
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    
    st.divider()
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# --- 4. MAIN LOGIC ---
st.title("🧠 DocuChat AI")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 5. SECURE KEY HANDLING ---
# This looks for keys in the "Secrets" vault, NOT in the code.
if "RAGIE_KEY" in st.secrets and "GOOGLE_KEY" in st.secrets:
    try:
        ragie = Ragie(auth=st.secrets["RAGIE_KEY"])
        genai.configure(api_key=st.secrets["GOOGLE_KEY"])
        
        # Auto-detect Model
        all_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        model = genai.GenerativeModel(all_models[0])

        if uploaded_file:
            if 'current_file' not in st.session_state or st.session_state.current_file != uploaded_file.name:
                with st.spinner("🧠 Reading document..."):
                    with open("temp.pdf", "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    with open("temp.pdf", "rb") as f:
                        doc = ragie.documents.create(request={"file": {"file_name": uploaded_file.name, "content": f}})
                    st.session_state['doc_id'] = doc.id
                    st.session_state.current_file = uploaded_file.name
                    st.toast("Ready!", icon="✅")

        if prompt := st.chat_input("Ask about your document..."):
            if 'doc_id' not in st.session_state:
                st.error("⚠️ Please upload a document first!")
            else:
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        results = ragie.retrievals.retrieve(request={"query": prompt, "filter": {"document_id": {"$eq": st.session_state['doc_id']}}})
                        context = "\n".join([chunk.text for chunk in results.scored_chunks])
                        response = model.generate_content(f"Context: {context}\n\nQuestion: {prompt}\n\nAnswer:")
                        st.markdown(response.text)
                st.session_state.messages.append({"role": "assistant", "content": response.text})

    except Exception as e:
        st.error(f"Error: {e}")
else:
    # This shows up if you forget to add secrets on the server
    st.error("⚠️ API Keys missing! Add them to Streamlit Secrets.")

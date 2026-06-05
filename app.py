import streamlit as st
import os
from groq import Groq
from rag_utils import process_uploaded_files, get_retriever

# ------------------------
# Page Config
# ------------------------
st.set_page_config(
    page_title="RAG Intelligence — Groq",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------
# Load CSS
# ------------------------
with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ------------------------
# Animated Header
# ------------------------
st.markdown("""
<div class="chat-header">
    <div class="header-logo">🤖</div>
    <div class="header-text">
        <h1>RAG Intelligence</h1>
        <p>Powered by Groq · LLaMA 3</p>
    </div>
    <div class="header-status">
        <span class="status-dot"></span>
        <span>Online</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ------------------------
# API Key Setup
# ------------------------
try:
    groq_api = st.secrets["GROQ_API_KEY"]
except Exception:
    groq_api = os.environ.get("GROQ_API_KEY", "")

if not groq_api:
    st.error("⚠️ **GROQ_API_KEY not found.** Add it to `.streamlit/secrets.toml` or set the environment variable.")
    st.stop()

client = Groq(api_key=groq_api)

# ------------------------
# Session State
# ------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm your AI assistant powered by Groq. Upload documents to enable **RAG mode**, or just start chatting!"}
    ]
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "doc_count" not in st.session_state:
    st.session_state.doc_count = 0

# ------------------------
# Sidebar
# ------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-header">📂 Knowledge Base</div>', unsafe_allow_html=True)

    model = st.selectbox(
        "Model",
        ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "mixtral-8x7b-32768"],
        help="Select the AI model"
    )

    st.divider()

    uploaded_files = st.file_uploader(
        "Upload Documents (PDF / TXT)",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="Upload documents to enable intelligent search"
    )

    if uploaded_files:
        with st.spinner("Indexing documents…"):
            try:
                st.session_state.vectorstore = process_uploaded_files(uploaded_files)
                st.session_state.doc_count = len(uploaded_files)
                st.success(f"✅ {len(uploaded_files)} document(s) indexed!")
            except Exception as e:
                st.error(f"Error processing files: {e}")

    if st.session_state.vectorstore:
        st.markdown(f"""
        <div class="rag-badge active">
            🟢 RAG Mode Active<br>
            <small>{st.session_state.doc_count} document(s) loaded</small>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="rag-badge inactive">
            ⚪ RAG Mode Inactive<br>
            <small>Upload documents to enable</small>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = [
                {"role": "assistant", "content": "Chat cleared! Ready to help."}
            ]
            st.rerun()
    with col2:
        if st.button("🔄 Reset All", use_container_width=True):
            st.session_state.messages = [
                {"role": "assistant", "content": "Everything reset! Upload documents or start chatting."}
            ]
            st.session_state.vectorstore = None
            st.session_state.doc_count = 0
            st.rerun()

    st.divider()
    st.markdown("""
    <div class="sidebar-footer">
        <p>💡 <strong>Tips</strong></p>
        <p>• Upload PDFs or TXT files</p>
        <p>• Ask questions about your docs</p>
        <p>• Switch models for different tasks</p>
    </div>
    """, unsafe_allow_html=True)

# ------------------------
# Chat History
# ------------------------
for msg in st.session_state.messages:
    avatar = "🤖" if msg["role"] == "assistant" else "👤"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# ------------------------
# Chat Input
# ------------------------
if prompt := st.chat_input("Ask me anything…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        placeholder = st.empty()

        # Build API messages
        system_msg = {"role": "system", "content": "You are a helpful, knowledgeable AI assistant. Provide clear, accurate, and concise responses. Format using markdown where appropriate."}

        if st.session_state.vectorstore:
            retriever = get_retriever(st.session_state.vectorstore)
            context_docs = retriever.invoke(prompt)
            context_text = "\n\n".join([d.page_content for d in context_docs])
            augmented = f"Context from uploaded documents:\n{context_text}\n\nUser question: {prompt}\n\nAnswer clearly using the context."
            api_messages = [system_msg] + [m for m in st.session_state.messages[:-1]] + [{"role": "user", "content": augmented}]
        else:
            api_messages = [system_msg] + st.session_state.messages

        # Stream response
        stream = client.chat.completions.create(
            model=model,
            messages=api_messages,
            stream=True,
            max_tokens=2048,
        )

        full_reply = ""
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                full_reply += delta
                placeholder.markdown(full_reply + "▌")

        placeholder.markdown(full_reply)

    st.session_state.messages.append({"role": "assistant", "content": full_reply})

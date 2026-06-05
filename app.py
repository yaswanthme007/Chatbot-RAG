import streamlit as st
import os
from groq import Groq
from rag_utils import process_uploaded_files, get_retriever

# ------------------------
# Page Config
# ------------------------
st.set_page_config(
    page_title="RAG Intelligence",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------
# Load CSS
# ------------------------
with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ------------------------
# Header
# ------------------------
st.markdown("""
<div class="chat-header">
    <div class="header-brand">
        <div class="brand-icon">◆</div>
        <div class="brand-text">
            <span class="brand-name">RAG Intelligence</span>
            <span class="brand-sub">Groq · LLaMA 3</span>
        </div>
    </div>
    <div class="header-status">
        <span class="status-wrap">
            <span class="status-ping"></span>
            <span class="status-dot"></span>
        </span>
        Online
    </div>
</div>
""", unsafe_allow_html=True)

# ------------------------
# API Key
# ------------------------
try:
    groq_api = st.secrets["GROQ_API_KEY"]
except Exception:
    groq_api = os.environ.get("GROQ_API_KEY", "")

if not groq_api:
    st.error("**GROQ_API_KEY not set.** Add it to `.streamlit/secrets.toml` or as an environment variable.")
    st.stop()

client = Groq(api_key=groq_api)

# ------------------------
# Session State
# ------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm your AI assistant powered by Groq.\n\nUpload a document in the sidebar to enable **RAG mode** — I'll answer questions grounded in your files. Or just start chatting."}
    ]
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "doc_count" not in st.session_state:
    st.session_state.doc_count = 0

# ------------------------
# Sidebar
# ------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-title">◆ &nbsp;RAG Intelligence</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section-label">Model</div>', unsafe_allow_html=True)
    model = st.selectbox(
        "Model",
        ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "mixtral-8x7b-32768"],
        label_visibility="collapsed",
        help="Select the AI model to use for responses"
    )

    st.divider()

    st.markdown('<div class="sidebar-section-label">Knowledge Base</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Upload Documents",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        help="Upload PDF or TXT files to enable document-grounded answers"
    )

    if uploaded_files:
        with st.spinner("Indexing…"):
            try:
                st.session_state.vectorstore = process_uploaded_files(uploaded_files)
                st.session_state.doc_count = len(uploaded_files)
                st.success(f"{len(uploaded_files)} document(s) indexed")
            except Exception as e:
                st.error(f"Failed to index: {e}")

    if st.session_state.vectorstore:
        st.markdown(f"""
        <div class="rag-card active">
            RAG Active
            <small>{st.session_state.doc_count} document(s) loaded</small>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="rag-card inactive">
            RAG Inactive
            <small>Upload documents to enable</small>
        </div>""", unsafe_allow_html=True)

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = [
                {"role": "assistant", "content": "Chat cleared. How can I help you?"}
            ]
            st.rerun()
    with col2:
        if st.button("Reset All", use_container_width=True):
            st.session_state.messages = [
                {"role": "assistant", "content": "Reset complete. Upload documents or start a new conversation."}
            ]
            st.session_state.vectorstore = None
            st.session_state.doc_count = 0
            st.rerun()

    st.divider()

    st.markdown("""
    <div class="tips-block">
        <p class="tips-heading">Quick Tips</p>
        <p>Upload PDFs or TXT files to enable document search</p>
        <p>Switch models based on speed vs quality needs</p>
        <p>RAG answers are grounded in your uploaded content</p>
    </div>
    """, unsafe_allow_html=True)

# ------------------------
# Chat History
# ------------------------
for msg in st.session_state.messages:
    avatar = "🤖" if msg["role"] == "assistant" else "🧑"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# ------------------------
# Chat Input
# ------------------------
if prompt := st.chat_input("Message RAG Intelligence…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        placeholder = st.empty()

        system_msg = {
            "role": "system",
            "content": (
                "You are a precise, helpful AI assistant. "
                "Provide clear and well-structured responses. "
                "Use markdown formatting — headers, bullet points, code blocks — where it aids clarity. "
                "Be concise but complete."
            )
        }

        if st.session_state.vectorstore:
            retriever = get_retriever(st.session_state.vectorstore)
            context_docs = retriever.invoke(prompt)
            context_text = "\n\n".join([d.page_content for d in context_docs])
            augmented = (
                f"Context from uploaded documents:\n{context_text}\n\n"
                f"User question: {prompt}\n\n"
                "Answer using the context above. If the context does not contain enough information, say so clearly."
            )
            api_messages = [system_msg] + [m for m in st.session_state.messages[:-1]] + [{"role": "user", "content": augmented}]
        else:
            api_messages = [system_msg] + st.session_state.messages

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

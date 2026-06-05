import streamlit as st
from groq import Groq

# ------------------------
# Page Config
# ------------------------
st.set_page_config(page_title="🤖 Groq Chatbot", layout="centered")
st.markdown("""
    <style>
        .main {
            background-color: #111827; /* dark background */
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .stChatMessage {
            border-radius: 18px;
            padding: 12px 16px;
            margin: 8px 0;
            font-size: 16px;
            line-height: 1.5;
            max-width: 75%;
            display: inline-block;
        }
        .stChatMessage.user {
            background-color: #3b82f6;  /* blue */
            color: #ffffff;             /* white text */
            text-align: right;
            margin-left: auto;
        }
        .stChatMessage.assistant {
            background-color: #f3f4f6;  /* light gray */
            color: #111827;             /* dark text */
            text-align: left;
            margin-right: auto;
        }
        .stButton button {
            border-radius: 8px;
            background-color: #2563eb;
            color: white;
            padding: 8px 16px;
            font-weight: bold;
        }
        .stButton button:hover {
            background-color: #1d4ed8;
        }
    </style>
""", unsafe_allow_html=True)

# ------------------------
# Title
# ------------------------
st.title("🤖 Groq Chatbot")
st.write("This chatbot is powered by the **Groq API** using Llama 3 models.")

# ------------------------
# API Key (hardcoded)
# ------------------------
import os
groq_api = os.environ.get("GROQ_API_KEY", "")

# Setup Groq client
client = Groq(api_key=groq_api)

# ------------------------
# Chat History
# ------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am your Groq-powered chatbot. How can I help you today?"}
    ]

# Display previous messages
for msg in st.session_state.messages:
    role_class = "user" if msg["role"] == "user" else "assistant"
    st.markdown(
        f'<div class="stChatMessage {role_class}">{msg["content"]}</div>',
        unsafe_allow_html=True
    )

# ------------------------
# Chat Input
# ------------------------
if prompt := st.chat_input("Type your message..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user input
    st.markdown(
        f'<div class="stChatMessage user">{prompt}</div>',
        unsafe_allow_html=True
    )

    # Generate response
    with st.spinner("Thinking..."):
        chat_completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",  # ✅ supported model
            messages=st.session_state.messages
        )
        reply = chat_completion.choices[0].message.content

    # Display assistant reply
    st.markdown(
        f'<div class="stChatMessage assistant">{reply}</div>',
        unsafe_allow_html=True
    )

    st.session_state.messages.append({"role": "assistant", "content": reply})

# ------------------------
# Clear Chat Button
# ------------------------
if st.button("🗑️ Clear Chat"):
    st.session_state.messages = [
        {"role": "assistant", "content": "Chat history cleared! How can I help you now?"}
    ]
    st.experimental_rerun()

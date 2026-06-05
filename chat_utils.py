import streamlit as st


def add_message(role: str, content: str):
    st.session_state.messages.append({"role": role, "content": content})

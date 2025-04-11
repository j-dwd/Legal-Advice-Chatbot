import streamlit as st

def check_authentication():
    if not st.session_state.get("authenticated", False):
        st.warning("Please log in to access the app.")
        return False
    return True
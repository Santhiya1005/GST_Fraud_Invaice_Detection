import streamlit as st
from supabase import create_client

st.title("Supabase Connection Test")

try:
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_SECRET_KEY"]

    client = create_client(url, key)
    client.table("invoice_analyses").select("id").limit(1).execute()

    st.success("Supabase connection successful!")
except Exception as e:
    st.error(f"Connection failed: {type(e).__name__}")
    st.code(str(e))
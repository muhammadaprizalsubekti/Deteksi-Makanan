import streamlit as st

st.title("🔐 Login NutriVision")

username = st.text_input("Username")
password = st.text_input(
    "Password",
    type="password"
)

if st.button("Login"):

    if username == "admin" and password == "123":

        st.session_state.login = True

        st.success("Login berhasil")

    else:

        st.error("Username atau Password salah")
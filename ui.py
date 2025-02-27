import streamlit as st
import requests

# Streamlit Page Configuration
st.set_page_config(page_title="Healthcare Q&A System", page_icon="💊")

st.title("💊 Healthcare Q&A System (Powered by Gemini GPT-4)")
st.write("Ask any medical-related question, and I will provide AI-generated responses.")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Input Field
user_input = st.chat_input("Type your medical question here...")

if user_input:
    # Add User Message to Chat History
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    # Send Question to Flask Backend
    with st.chat_message("assistant"):
        with st.spinner("Fetching response..."):
            try:
                response = requests.post("http://localhost:5000/ask", json={"question": user_input})
                answer = response.json().get("answer", "Sorry, I couldn't fetch a response.")
            except:
                answer = "Error: Could not connect to the server."

            st.markdown(answer)

    # Add AI Response to Chat History
    st.session_state.messages.append({"role": "assistant", "content": answer})

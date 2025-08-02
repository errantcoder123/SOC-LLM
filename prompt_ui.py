from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st

# Load environment variables (e.g., API keys)
load_dotenv()

# Initialize the Gemini model
genai_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# Streamlit UI setup
st.title("Academic Summary Assistant")

query = st.text_input("Type your research prompt here:")

if st.button("Generate Summary"):
    response = genai_model.invoke(query)
    st.subheader("Summary Output:")
    st.write(response.content)

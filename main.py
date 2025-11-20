
import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
load_dotenv()
import os
hf_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")

try:
    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        task="text-generation",
        huggingfacehub_api_token=hf_key,
    )
    model = ChatHuggingFace(llm=llm)
except Exception as e:
    st.error(f"Model initialization failed: {e}")
    st.stop()

# ---------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------
st.title("🧠 Research Tool (SSL bypass mode)")

user_input = st.text_area("Enter your query:")

if st.button("Generate Response"):
    if not user_input.strip():
        st.warning("Please enter a query.")
    else:
        with st.spinner("Generating response..."):
            try:
                result = model.invoke(user_input)
                st.success("Response:")
                st.write(result.content)
            except Exception as e:
                st.error(f"⚠️ Error during generation: {e}")

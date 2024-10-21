import streamlit as st
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from langchain.schema import Document
from operator import itemgetter
import textwrap
from typing import List
from qdrant_client import QdrantClient
from langchain_core.documents import Document
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings, HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import GoogleGenerativeAI
import os
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import time
import logging

# Load environment variables
load_dotenv()

# Initialize Qdrant vector store
qdrant_url = os.getenv("QDRANT_URL")
qdrant_key = os.getenv("QDRANT_API_KEY")
collection_name = "medication_interaction"
embed_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")  # Replace with your embedding model

# Configure logging
logging.basicConfig(level=logging.DEBUG)

try:
    qdrant = QdrantVectorStore.from_existing_collection(
        embedding=embed_model,
        collection_name=collection_name,
        url=qdrant_url,
        api_key=qdrant_key,
        prefer_grpc=True,
        timeout=10.0
    )
except Exception as e:
    logging.error(f"Failed to connect to Qdrant: {e}")
    st.error("Failed to connect to Qdrant. Please check the server and try again.")
    st.stop()

def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

def format_output(text: str) -> str:
    paragraphs = text.split('\n')
    formatted_paragraphs = [textwrap.fill(p.strip(), width=80) for p in paragraphs if p.strip()]
    return '\n\n'.join(formatted_paragraphs)

# Define the prompt template
prompt_str = """
Answer the user question based on your knowledge and the following context:
{context}

Question: {question}
Act as a medical practitioner. Analyze drug-to-drug interactions and provide a clear and concise answer.
Format your response in easy-to-read paragraphs.
Include a numbered list of specific drug interactions.
If possible, mention the mechanism of interaction and any recommended monitoring or precautions.
"""
_prompt = ChatPromptTemplate.from_template(prompt_str)

# Set up the retriever from your existing Qdrant vector store
num_chunks = 5  # Increased from 3 to potentially get more comprehensive information
retriever = qdrant.as_retriever(search_type="similarity", search_kwargs={"k": num_chunks})

# Set up the Google Generative AI model
google_api_key = os.getenv("GOOGLE_API_KEY")  # Replace with your Google API key
llm_name = "gemini-1.5-flash"  # Replace with the name of the LLM you are using
chat_llm = GoogleGenerativeAI(model=llm_name, google_api_key=google_api_key, temperature=0.3)

# Define the query fetcher
query_fetcher = itemgetter("question")

# Set up the chain
setup = {
    "question": query_fetcher,
    "context": lambda x: format_docs(retriever.get_relevant_documents(x["question"]))
}
_chain = setup | _prompt | chat_llm

def get_drug_interactions(drug_name: str) -> str:
    query = f"What are the drug-to-drug interactions of {drug_name}?"
    response = _chain.invoke({"question": query})
    formatted_response = format_output(response)
    return f"Query: {query}\n\nResponse:\n\n{formatted_response}"

# Streamlit application
def main():
    st.set_page_config(page_title="Clinical Decision Support System", page_icon=":pill:")
    st.title("Welcome to Clinical Decision Support System")
    st.balloons()
    st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT1_QBFhVJQz_9ncP3evnEyQHU4XWxcBWgMIw&s", use_column_width=True)

    # Sidebar
    st.sidebar.header("About")
    st.sidebar.text("This project developed by Ali Imran")
    contact = st.sidebar.selectbox("Contact Us", ["Email: imrankhan.ai.engineer@gmail.com", "Facebook: Ai Tech", "Twitter: @imrankhan_ai"])
    if "drug_name_input" not in st.session_state:
        st.session_state["drug_name_input"] = ""
    # User input for drug name
    drug_name = st.text_input("Enter the name of the drug:", key="drug_name_input", placeholder="e.g., Ibuprofen")

    # Button to submit the query
    if st.button("Get Drug Interactions"):
        if drug_name:
            with st.spinner('Fetching drug interactions...'):
                try:
                    response = get_drug_interactions(drug_name)
                    st.session_state["response"] = response  # Store the response in session state
                    st.experimental_rerun()  # Re-run the app to clear the input field
                except Exception as e:
                    logging.error(f"Successfully to get drug interactions: {e}")
                    st.warning("Successfully to get drug interactions. Please try a more generic name of medicine query")
        else:
            st.warning("Please enter a drug name.")

    # Display the response if it exists
    if "response" in st.session_state:
        st.text_area("Response", st.session_state["response"], height=300)

if __name__ == "__main__":
    main()
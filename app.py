import streamlit as st
import logging
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
from autogen.agentchat.contrib.retrieve_assistant_agent import RetrieveAssistantAgent
import requests
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import faiss
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

# Set up the title of the Streamlit app
st.title("Meal Plan Assistant")

# Load API keys from Streamlit secrets
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
EDAMAM_APP_KEY = st.secrets["EDAMAM_APP_KEY"]
EDAMAM_APP_ID = st.secrets.get("EDAMAM_APP_ID", "fcbbd9b3")  # Replace with your actual Edamam App ID if different

# Configure OpenAI
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

config_list = [{"model": "gpt-3.5-turbo", "api_key": OPENAI_API_KEY}]

# Define the Edamam API agent to retrieve recipes
class EdamamAPIAgent:
    def __init__(self, app_id, app_key):
        self.app_id = app_id
        self.app_key = app_key

    def search_recipes(self, query, health_labels=None, diet_labels=None, max_results=10):
        endpoint = "https://api.edamam.com/api/recipes/v2"
        params = {
            "type": "public",
            "q": query,
            "app_id": self.app_id,
            "app_key": self.app_key,
            "to": max_results
        }
        if health_labels:
            params["health"] = health_labels
        if diet_labels:
            params["diet"] = diet_labels
        try:
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"Edamam API request failed: {e}")
            return {"error": str(e)}

# Instantiate the Edamam API agent
edamam_agent = EdamamAPIAgent(EDAMAM_APP_ID, EDAMAM_APP_KEY)

# Initialize LangChain's OpenAI Embeddings
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Function to load documents and create FAISS vector store
def load_faiss_vector_store(docs_path, embedding_model):
    all_documents = []
    for doc_url in docs_path:
        try:
            response = requests.get(doc_url)
            response.raise_for_status()
            content = response.text
            # Simple split into chunks; consider using a more robust method for large documents
            chunks = content.split('\n\n')  # Split by paragraphs
            for chunk in chunks:
                all_documents.append({"content": chunk})
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to fetch document from {doc_url}: {e}")
    
    # Convert documents to list of strings
    document_texts = [doc["content"] for doc in all_documents if doc["content"].strip() != ""]
    
    # Create FAISS vector store
    vector_store = FAISS.from_texts(document_texts, embeddings)
    return vector_store

# Load documents and create FAISS vector store
docs_path = [
    "https://www.nhlbi.nih.gov/sites/default/files/publications/WeekOnDASH.pdf",
    "https://www.nhlbi.nih.gov/files/docs/public/heart/new_dash.pdf",
]

vector_store = load_faiss_vector_store(docs_path, embedding_model=config_list[0]["model"])

# Create the main assistant agent for meal planning
assistant = RetrieveAssistantAgent(
    name="MealPlanAssistant",
    system_message='''
    You are a helpful meal planning assistant. Greet the user, ask for their information (name, zip code, chronic disease, cuisine preference, and ingredient dislikes). Tailor the meal plan based on the customer's chronic disease and preferences, and use the Edamam API to find specific recipes that match the customer's needs.
    ''',
    llm_config={"config_list": config_list}
)

# Create a RetrieveUserProxyAgent for document-based retrieval using FAISS
def label_rag_response(response):
    return {"source": "RAG System (PDFs)", "data": response}

ragproxyagent = RetrieveUserProxyAgent(
    name="UserProxy",
    human_input_mode="ALWAYS",
    max_consecutive_auto_reply=1,
    retrieve_config={
        "task": "qa",
        "chunk_token_size": 1000,
        "model": config_list[0]["model"],
        "chunk_mode": "multi_lines",
        "custom_callback": label_rag_response,
        "vector_store": vector_store  # Pass the FAISS vector store
    },
    llm_config={"config_list": config_list},
    function_map={"search_recipes": edamam_agent.search_recipes}
)

# Function to initiate the chat
def start_chat(user_message):
    # Start or continue chat
    ragproxyagent.initiate_chat(
        assistant,
        message=user_message
    )

# Session state to store the chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input and process it
user_input = st.chat_input("You: ")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    # Process the response and initiate the chat
    start_chat(user_input)

    # Retrieve the assistant's response
    # Note: Ensure that `RetrieveAssistantAgent` has a method to get the last message
    assistant_response = assistant.last_message(ragproxyagent)['content']

    st.session_state.messages.append({"role": "assistant", "content": assistant_response})

    with st.chat_message("assistant"):
        st.markdown(assistant_response)

# Button to clear chat history
if st.button("Clear Chat History"):
    st.session_state.messages = []
    st.experimental_rerun()

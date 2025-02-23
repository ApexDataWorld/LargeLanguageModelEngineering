import os
import json
import requests
import streamlit as st
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool

# Document Processor
class DocumentProcessor:
    def __init__(self, chunk_size=500, chunk_overlap=50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_documents(self, file_paths):
        documents = []
        for file_path in file_paths:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            loader = TextLoader(file_path)
            documents.extend(loader.load())
        return documents

    def split_documents(self, documents):
        text_splitter = CharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        return text_splitter.split_documents(documents)

    def create_vector_store(self, documents):
        embeddings = HuggingFaceEmbeddings()
        return FAISS.from_documents(documents, embeddings)

# Chain Builder
class ChainBuilder:
    def __init__(self, model_name="gpt-4", temperature=0.7):
        self.model_name = model_name
        self.temperature = temperature

    def build_chain(self, vector_store):
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        llm = OpenAI(model_name=self.model_name, temperature=self.temperature)
        return ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(),
            memory=memory,
            return_source_documents=True
        )

# Tools
def fetch_stock_price(symbol):
    API_KEY = "your_alpha_vantage_api_key"
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data.get("Global Quote", {}).get("05. price", "Price not found.")
    return "Error fetching stock price."

def sentiment_analysis(text):
    return "Positive" if "good" in text else "Negative"

stock_price_tool = Tool(
    name="Stock Price Fetcher",
    func=fetch_stock_price,
    description="Fetch real-time stock prices for a given symbol."
)

sentiment_tool = Tool(
    name="Sentiment Analyzer",
    func=sentiment_analysis,
    description="Perform sentiment analysis on input text."
)

# Feedback Logger
class FeedbackLogger:
    def __init__(self, log_file="feedback_log.json"):
        self.log_file = log_file

    def log_feedback(self, user_input, assistant_response, feedback):
        log_entry = {
            "user_input": user_input,
            "assistant_response": assistant_response,
            "feedback": feedback
        }
        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

# Streamlit App
st.set_page_config(page_title="Conversational Document Assistant", layout="wide")
st.title("📄 Conversational Document Assistant")

doc_processor = DocumentProcessor()
chain_builder = ChainBuilder()
feedback_logger = FeedbackLogger()

uploaded_files = st.file_uploader("Upload documents (TXT, CSV, or PDF)", type=["txt", "csv", "pdf"], accept_multiple_files=True)

if "chain" not in st.session_state:
    st.session_state.chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if uploaded_files and st.button("Process Documents"):
    try:
        with st.spinner("Processing documents..."):
            file_paths = [uploaded_file.name for uploaded_file in uploaded_files]
            documents = doc_processor.load_documents(file_paths)
            split_docs = doc_processor.split_documents(documents)
            vector_store = doc_processor.create_vector_store(split_docs)
            st.session_state.chain = chain_builder.build_chain(vector_store)
            st.success("Documents processed successfully!")
    except Exception as e:
        st.error(f"Error: {e}")

if st.session_state.chain:
    user_input = st.text_input("Ask a question about the documents or use tools:")
    if user_input:
        with st.spinner("Generating response..."):
            response = st.session_state.chain({"question": user_input})
            st.session_state.chat_history.append((user_input, response["answer"]))

        for user_query, bot_response in st.session_state.chat_history:
            st.markdown(f"**You**: {user_query}")
            st.markdown(f"**Assistant**: {bot_response}")

        feedback = st.selectbox("Was this response helpful?", ["Yes", "No", "Somewhat"])
        if st.button("Submit Feedback"):
            feedback_logger.log_feedback(user_input, response["answer"], feedback)
            st.success("Feedback submitted!")

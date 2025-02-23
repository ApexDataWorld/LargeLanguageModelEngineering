import streamlit as st
from document_processor import DocumentProcessor
from chain_builder import ChainBuilder
from tools import stock_price_tool, sentiment_tool
from feedback_logger import FeedbackLogger
import os
import glob

# Debug: Print the current working directory
print("Current working directory:", os.getcwd())

# Streamlit setup
st.set_page_config(page_title="Conversational Document Assistant", layout="wide")
st.title("📄 Conversational Document Assistant")

# Initialize session state attributes
if "chain" not in st.session_state:
    st.session_state.chain = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "local_files" not in st.session_state:
    st.session_state.local_files = []

# Initialize components
doc_processor = DocumentProcessor()
chain_builder = ChainBuilder()
feedback_logger = FeedbackLogger()

# Specify the path to the local temp folder
TEMP_FOLDER_PATH = "/Users/saurabhgupta/projects/ML/conversational_document_assistant/documents"

# Load files from the local temp folder
if st.button("Load Local Documents"):
    try:
        with st.spinner("Loading documents from local temp folder..."):
            # Find all supported files in the temp folder
            file_paths = glob.glob(os.path.join(TEMP_FOLDER_PATH, "*"))
            supported_files = [f for f in file_paths if f.endswith(('.txt', '.csv', '.pdf'))]
            
            if not supported_files:
                st.warning("No supported documents found in the temp folder!")
            else:
                st.success(f"Found {len(supported_files)} document(s): {supported_files}")
                st.session_state.local_files = supported_files
    except Exception as e:
        st.error(f"Error: {e}")

# Process local files
if "local_files" in st.session_state and st.button("Process Local Documents"):
    try:
        with st.spinner("Processing documents..."):
            documents = doc_processor.load_documents(st.session_state.local_files)
            split_docs = doc_processor.split_documents(documents)
            vector_store = doc_processor.create_vector_store(split_docs)
            st.session_state.chain = chain_builder.build_chain(vector_store)
            st.success("Local documents processed successfully!")
    except Exception as e:
        st.error(f"Error: {e}")

# Chat interface
if st.session_state.chain:
    user_input = st.text_input("Ask a question about the documents or use tools:")
    if user_input:
        with st.spinner("Generating response..."):
            #response = st.session_state.chain({"question": user_input})
            #response = chain_builder.answer_question(user_input, st.session_state.chain)

            #st.session_state.chat_history.append((user_input, response["answer"]))

            response = chain_builder.answer_question(user_input, st.session_state.chain)

            # Ensure response is correctly formatted
            answer = response.get("answer", "Sorry, I couldn't generate an answer.")

            st.session_state.chat_history.append((user_input, answer))


        # Display chat history
        for user_query, bot_response in st.session_state.chat_history:
            st.markdown(f"**You**: {user_query}")
            st.markdown(f"**Assistant**: {bot_response}")

        # Feedback
        feedback = st.selectbox("Was this response helpful?", ["Yes", "No", "Somewhat"])
        if st.button("Submit Feedback"):
            feedback_logger.log_feedback(user_input, response["answer"], feedback)
            st.success("Feedback submitted!")

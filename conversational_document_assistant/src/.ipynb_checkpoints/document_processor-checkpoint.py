import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import numpy as np



class DocumentProcessor:
    def __init__(self, chunk_size=500, chunk_overlap=50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_documents(self, file_paths):
        """Load documents from the provided file paths."""
        documents = []
        for file_path in file_paths:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            loader = TextLoader(file_path)  # Use the updated TextLoader
            documents.extend(loader.load())
        return documents

    def split_documents(self, documents):
        """Split documents into smaller chunks."""
        text_splitter = CharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        split_docs = text_splitter.split_documents(documents)
        return split_docs

    def create_vector_store(self, documents):

        # Retrieve API key from environment variables
        api_key = os.getenv("OPENAI_API_KEY")  

        # Initialize OpenAI Embeddings
        embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            openai_api_key=api_key  # Use the retrieved API key
        )
        doc_texts = [doc.page_content for doc in documents]
        vector_store = FAISS.from_texts(doc_texts, embeddings)
        return vector_store

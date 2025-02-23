from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

class ChainBuilder:
    def __init__(self, model_name="gpt-4", temperature=0.7):
        self.model_name = model_name
        self.temperature = temperature

    def build_chain(self, vector_store):
        memory = ConversationBufferMemory(
            memory_key="chat_history",  
            return_messages=True,       
            output_key="answer"  # Store only the answer in memory
        )

        llm = ChatOpenAI(model=self.model_name, temperature=self.temperature)

        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(),
            memory=memory,
            return_source_documents=True,
            output_key="answer"
        )

        return chain

    def answer_question(self, query, chain):
        """Handles user queries, including those outside document knowledge."""
        response = chain({"question": query})

        # If response from the documents is empty, use general LLM (GPT-4)
        if not response.get("answer", "").strip():  
            general_llm = ChatOpenAI(model_name=self.model_name, temperature=self.temperature)
            general_response = general_llm.predict(query)

            return {"answer": general_response}  # Ensure response is a dictionary

        return response  # Return document-based response normally

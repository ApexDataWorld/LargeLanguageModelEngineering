from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

class ChainBuilder:
    def __init__(self, model_name="gpt-4", temperature=0.7):
        self.model_name = model_name
        self.temperature = temperature

    def build_chain(self, vector_store):
        memory = ConversationBufferMemory(
            memory_key="chat_history",  # Stores previous conversations
            return_messages=True,       # Ensures messages are retrieved properly
            output_key="answer"         # Explicitly tell memory to store  "answer"
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

        if response["answer"].strip():  # If the model finds an answer from documents
            return response["answer"]
        else:  # If the model doesn't find an answer, use GPT-4 directly
            general_llm = ChatOpenAI(model_name=self.model_name, temperature=self.temperature)
            general_response = general_llm.predict(query)
            return general_response

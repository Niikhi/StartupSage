from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.schema import (
    BaseMessage,
    SystemMessage,
    HumanMessage,
    AIMessage
)
from langchain_core.memory import BaseMemory
from typing import Dict, Any, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from pydantic import Field

class SummarizingChatMessageHistory:
    messages: List[BaseMessage]
    llm: Any
    max_tokens: int
    summarize_interval: int
    summary: str

    def __init__(self, llm, max_tokens=1000, summarize_interval=10):
        self.messages = []
        self.llm = llm
        self.max_tokens = max_tokens
        self.summarize_interval = summarize_interval
        self.summary = ""

    def add_message(self, message: BaseMessage) -> None:
        self.messages.append(message)
        if len(self.messages) % self.summarize_interval == 0:
            self.summarize()

    def summarize(self):
        if len(self.messages) <= self.summarize_interval:
            return

        to_summarize = self.messages[:-self.summarize_interval]
        recent = self.messages[-self.summarize_interval:]

        docs = [Document(page_content=m.content) for m in to_summarize]
        chain = load_summarize_chain(self.llm, chain_type="map_reduce")
        summary = chain.run(docs)

        self.summary = summary
        self.messages = [SystemMessage(content=f"Summary of earlier conversation: {summary}")] + recent

    def get_messages(self):
        return [SystemMessage(content=self.summary)] + self.messages if self.summary else self.messages

    def clear(self):
        self.messages = []
        self.summary = ""

class SummarizingConversationBufferMemory(BaseMemory):
    memory: SummarizingChatMessageHistory = Field(default_factory=lambda: None)
    return_messages: bool = Field(default=False)
    input_key: str = Field(default="input")
    output_key: str = Field(default="output")

    def __init__(self, llm, return_messages: bool = False, input_key: str = "input", output_key: str = "output"):
        super().__init__(
            memory=SummarizingChatMessageHistory(llm),
            return_messages=return_messages,
            input_key=input_key,
            output_key=output_key
        )

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        input_str = inputs.get(self.input_key, "")
        output_str = outputs.get(self.output_key, "")
        
        self.memory.add_message(HumanMessage(content=input_str))
        self.memory.add_message(AIMessage(content=output_str))

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if self.return_messages:
            return {"history": self.memory.get_messages()}
        else:
            return {"history": self._get_chat_string()}

    def _get_chat_string(self) -> str:
        messages = self.memory.get_messages()
        return "\n".join([f"{m.type}: {m.content}" for m in messages])

    def clear(self) -> None:
        self.memory.clear()

    @property
    def memory_variables(self) -> List[str]:
        return ["history"]



# from flask import Blueprint, request, jsonify
# import time
# from langchain_groq import ChatGroq
# from langchain_community.vectorstores import Chroma
# from langchain_core.output_parsers import StrOutputParser
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.chains import LLMChain
# from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder
# from .summarize import SummarizingConversationBufferMemory

# main = Blueprint('main', __name__)

# @main.route('/query', methods=['POST'])
# def query():
#     data = request.get_json()
#     user_question = data.get('question')

#     embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
#     embedding_function = HuggingFaceEmbeddings(model_name=embedding_model_name)

#     chromadb_index = Chroma(persist_directory='./chroma_db', embedding_function=embedding_function)
#     retriever = chromadb_index.as_retriever()

#     groq_api_key = "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"
#     llm = ChatGroq(groq_api_key=groq_api_key, model_name='llama3-8b-8192')

#     memory = SummarizingConversationBufferMemory(llm, return_messages=True)

#     docs = retriever.get_relevant_documents(user_question)
#     context = "\n".join([doc.page_content for doc in docs])

#     prompt = ChatPromptTemplate.from_messages([
#         SystemMessagePromptTemplate.from_template(
#             "You are a startup-focused chatbot. Use the following context to answer the user's question: {context}"
#         ),
#         MessagesPlaceholder(variable_name="history"),
#         HumanMessagePromptTemplate.from_template("{input}")
#     ])

#     chain = LLMChain(
#         llm=llm,
#         prompt=prompt,
#         memory=memory,
#         output_key="output"
#     )

#     start_time = time.time()
#     response = chain.run(input=user_question, context=context)
    
#     end_time = time.time()

#     response_time = f"Response time: {end_time - start_time:.2f} seconds."
#     full_response = f"{response}\n\n{response_time}"

#     return jsonify({'response': full_response})
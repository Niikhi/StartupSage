# from flask import Blueprint, request, jsonify
# import time
# from typing import Dict, Any, List
# from langchain_groq import ChatGroq
# from langchain_community.vectorstores import Chroma
# from langchain_core.output_parsers import StrOutputParser
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.chains import LLMChain
# from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder
# from .history import CustomConversationBufferMemory

# main = Blueprint('main', __name__)

# memory = CustomConversationBufferMemory(return_messages=True, max_messages=10)

# @main.route('/query', methods=['POST'])
# def query():
#     data = request.get_json()
#     user_question = data.get('question')

#     embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
#     embedding_function = HuggingFaceEmbeddings(model_name=embedding_model_name)

#     chromadb_index = Chroma(persist_directory='./chroma_db', embedding_function=embedding_function)
#     retriever = chromadb_index.as_retriever(search_kwargs={"k": 5})  

#     groq_api_key = "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"
#     llm = ChatGroq(
#         groq_api_key=groq_api_key, 
#         model_name='llama3-8b-8192',
#         temperature=0.2,  
#         max_tokens=300 
#         )

#     docs = retriever.get_relevant_documents(user_question)
#     context = "\n".join([f"Document {i+1}:\n{doc.page_content}\n" for i, doc in enumerate(docs)])

#     system_template = """You are a startup-focused chatbot. Your knowledge comes EXCLUSIVELY from the following context. 
#                         Do not use any information outside of this context. If the context doesn't contain relevant information, 
#                         say that you don't have enough information to answer the question.

#                         Context:
#                         {context}

#                         Answer the user's question based ONLY on the information provided in the context above. 
#                         If the question cannot be answered using the given context, say so clearly."""

#     prompt = ChatPromptTemplate.from_messages([
#         SystemMessagePromptTemplate.from_template(system_template),
#         MessagesPlaceholder(variable_name="history"),
#         HumanMessagePromptTemplate.from_template("{input}")
#     ])

#     chain = LLMChain(
#         llm=llm,
#         prompt=prompt,
#         memory=memory,
#         output_parser=StrOutputParser()
#     )

#     start_time = time.time()
#     response = chain.run(input=user_question, context=context)
    
#     end_time = time.time()

#     response_time = f"Response time: {end_time - start_time:.2f} seconds."
#     full_response = f"{response}\n\n{response_time}"

#     return jsonify({
#         'response': full_response,
#         'context_used': context, 
#         'num_docs_retrieved': len(docs)
#     })

import os
from flask import Blueprint, request, jsonify
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder
import logging
from .history import CustomConversationBufferMemory
from .setup_neo4j import check_fulltext_index
from .query_operation import Neo4jOperations

check_fulltext_index()
main = Blueprint('main', __name__)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

neo4j_ops = Neo4jOperations()

memory = CustomConversationBufferMemory(return_messages=True, max_messages=20)



@main.route('/query', methods=['POST'])
async def query():
    data = request.get_json()
    user_question = data.get('question')

    neo4j_ops = Neo4jOperations()
    results = await neo4j_ops.query_neo4j_async(user_question)

    # Format the results
    # formatted_context = "Here are the top startup results based on the query:\n\n"
    # for entity_type, results in neo4j_results.items():
    #     formatted_context += f"Results for {entity_type}:\n"
    #     for result in results:
    #         formatted_context += f"Name: {result['name']}\n"
    #         formatted_context += f"Location: {result['location']}\n"
    #         formatted_context += f"Category: {result['category']}\n"
    #         formatted_context += f"Status: {result['status']}\n\n"
    #     formatted_context += "\n"

    formatted_results = []
    for result in results:
        formatted_results.append(str(result))

    groq_api_key = os.getenv("GROQ_API_KEY")
    llm = ChatGroq(
        groq_api_key=groq_api_key, 
        model_name='llama3-8b-8192', 
        temperature=0,
        max_tokens=300
    )

    system_template = """You are a startup-focused chatbot. Your knowledge comes EXCLUSIVELY from the following context. 
                        Do not use any information outside of this context. If the context doesn't contain relevant information, 
                        say that you don't have enough information to answer the question.

                        Context:
                        {context}

                        Answer the user's question based ONLY on the information provided in the context above. 
                        If asked about specific details like locations, always provide the full information available in the context.
                        If the question cannot be answered using the given context, say so clearly."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])

    chain = LLMChain(
        llm=llm, 
        prompt=prompt,
        memory=memory,
        verbose=True,
        output_parser=StrOutputParser()
    )

    try:
        response = chain.run(input=user_question, context=formatted_results)
    except Exception as e:
        print(f"Error in LLMChain: {str(e)}")
        response = "I'm sorry, but I encountered an error while processing your request."

    full_response = f"AI Assistant: {response}\n"

    return jsonify({
        'response': full_response,
        'context_used': formatted_results,
    })

@main.teardown_app_request
def close_neo4j_driver(exception=None):
    neo4j_ops.close()
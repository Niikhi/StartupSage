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
import asyncio 
from flask import Blueprint, request, jsonify, current_app
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
import logging
from .history import CustomConversationBufferMemory
from .setup_neo4j import check_fulltext_index
from .query_operation import Neo4jOperations
from langchain.schema import HumanMessage
import textwrap

check_fulltext_index()
main = Blueprint('main', __name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

memory = CustomConversationBufferMemory(return_messages=True, max_messages=5)

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name='llama3-8b-8192',
    temperature=0,
    max_tokens=300
)

system_template = """You are a startup-focused chatbot. Your knowledge comes EXCLUSIVELY from the provided context about startups. 
                    Do not use any information outside of this context. If the context doesn't contain relevant information, 
                    say that you don't have enough information to answer the question, but offer to help with other startup-related queries.

                    When answering:
                    1. If the context contains relevant information, refer to specific startups and their details.
                    2. If asked about a category or type of startup, list relevant startups from the context if available.
                    3. If no relevant information is found in the context, clearly state that you don't have that information.
                    4. If the question is a greeting or small talk, respond in a friendly manner, but don't provide any startup information unless it's in the context.
                    5. Maintain continuity in the conversation by referencing previous questions and answers when appropriate.
                    6. Only provide information that is explicitly stated in the given context or previous conversation history.
                    7. For follow-up questions, refer to the conversation history to provide consistent and contextual responses.

                    Always base your responses solely on the information provided in the context and the conversation history."""

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

def format_results(results):
    formatted = []
    for result in results:
        startup_info = {
            "name": result.get("s.name", "Unknown"),
            "category": result.get("c.name", "Unknown"),
            "location": result.get("l.name", "Unknown"),
            "status": result.get("st.name", "Unknown"),
            "relevance": result.get("s.relevance", "Unknown")
        }
        formatted.append(startup_info)
    return formatted

@main.route('/query', methods=['POST'])
async def query():
    logger.info("Received a request to /query endpoint")
    data = request.get_json()
    user_question = data.get('question')
    logger.info(f"Received question: {user_question}")

    try:
        if not current_app.neo4j_ops or not current_app.neo4j_ops.is_connected:
            logger.error("Not connected to Neo4j database")
            return jsonify({'response': "I'm sorry, but I'm currently unable to access the database. Please try again later or contact support."})

        results = await current_app.neo4j_ops.query_neo4j_async(user_question)
        
        if not results:
            context = "No specific startup information found in the database."
            formatted_results = []
        else:
            formatted_results = results  # The results are already formatted from query_neo4j_async
            logger.info(f"Formatted results: {formatted_results}")
            context = "Here are some relevant startups:\n" + "\n".join([f"- {r['name']} (Location: {r['location']}, Category: {r['category']}, Status: {r['status']})" for r in formatted_results])

        # Truncate context if it's too long
        max_context_length = 2000  # Adjust this value as needed
        if len(context) > max_context_length:
            context = textwrap.shorten(context, width=max_context_length, placeholder="...")
        
        history = memory.chat_memory.messages
        history_str = "\n".join([f"{'Human' if isinstance(msg, HumanMessage) else 'AI'}: {msg.content}" for msg in history])
        
        # Combine the context and user question
        combined_input = f"Context: {context}\n\nConversation History:\n{history_str}\n\nUser Question: {user_question}"

        response = await chain.ainvoke({
            "input": combined_input
        })

        generated_text = response['text']

        logger.info(f"LLM response: {generated_text}")
        full_response = f"AI Assistant: {generated_text}\n"
        
        memory.save_context({"input": user_question}, {"output": generated_text})

        return jsonify({
            'response': full_response,
            'context_used': context,
            'startups': formatted_results 
        })
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        return jsonify({
            'error': 'An error occurred while processing your request.',
            'details': str(e)
        }), 500

@main.teardown_app_request
def close_neo4j_driver(exception=None):
    neo4j_ops = current_app.neo4j_ops
    if neo4j_ops and neo4j_ops.is_connected:
        asyncio.run(neo4j_ops.close())
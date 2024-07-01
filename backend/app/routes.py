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
from dotenv import load_dotenv
from flask import Blueprint, request, jsonify
from neo4j import GraphDatabase
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
import spacy
import time

load_dotenv()

main = Blueprint('main', __name__)

URI = os.getenv("NEO4J_URI")
USER = os.getenv("NEO4J_USER")
PASSWORD = os.getenv("NEO4J_PASSWORD") 

driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

nlp = spacy.load("en_core_web_sm")

def extract_entities(question):
    doc = nlp(question)
    entities = {
        'LOCATION': [],
        'ORG': [],
        'PRODUCT': [],
        'GPE': [] 
    }
    for ent in doc.ents:
        if ent.label_ in entities:
            entities[ent.label_].append(ent.text)
    return entities

def construct_cypher_query(entities, question):
    query_parts = []
    params = {}

    if entities['ORG'] or entities['PRODUCT']:
        query_parts.append("MATCH (s:Startup)")
        if entities['ORG']:
            query_parts.append("WHERE s.name =~ $org_regex")
            params['org_regex'] = f"(?i).*{'|'.join(entities['ORG'])}.*"
        if entities['PRODUCT']:
            query_parts.append("WHERE s.category =~ $product_regex OR EXISTS((s)-[:IN_CATEGORY]->(:Category {name: $product}))")
            params['product_regex'] = f"(?i).*{'|'.join(entities['PRODUCT'])}.*"
            params['product'] = entities['PRODUCT'][0].lower()

    if entities['LOCATION'] or entities['GPE']:
        if not query_parts:
            query_parts.append("MATCH (s:Startup)")
        query_parts.append("WHERE s.location =~ $location_regex")
        params['location_regex'] = f"(?i).*{'|'.join(entities['LOCATION'] + entities['GPE'])}.*"

    if not query_parts:
        
        query_parts.append("MATCH (s:Startup)")
        query_parts.append("WHERE s.name =~ $general_regex OR s.category =~ $general_regex OR s.location =~ $general_regex OR s.status =~ $general_regex")
        params['general_regex'] = f"(?i).*{question}.*"

    query_parts.append("RETURN s.name AS name, s.location AS location, s.category AS category, s.status AS status")
    query_parts.append("LIMIT 5")

    return " ".join(query_parts), params

def query_neo4j(question):
    entities = extract_entities(question)
    query, params = construct_cypher_query(entities, question)

    with driver.session() as session:
        result = session.run(query, params)
        
        startup_info = []
        for record in result:
            info = f"Name: {record['name']}\n"
            info += f"Location: {record['location']}\n"
            info += f"Category: {record['category']}\n"
            info += f"Status: {record['status']}"
            startup_info.append(info)
        
        return "\n\n".join(startup_info)

@main.route('/query', methods=['POST'])
def query():
    data = request.get_json()
    user_question = data.get('question')

    context = query_neo4j(user_question)

    groq_api_key = os.getenv("GROQ_API_KEY")
    llm = ChatGroq(groq_api_key=groq_api_key, model_name='llama3-8b-8192', temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "You are a startup-focused chatbot. Your knowledge comes EXCLUSIVELY from the following context. "
            "Do not use any information outside of this context. If the context doesn't contain relevant information, "
            "say that you don't have enough information to answer the question.\n\nContext:\n{context}\n\n"
            "Answer the user's question based ONLY on the information provided in the context above. "
            "If you can't find the answer in the context, say so clearly."
        ),
        HumanMessagePromptTemplate.from_template("{input}")
    ])

    chain = LLMChain(llm=llm, prompt=prompt)

    
    response = "AI Assistant: " + chain.run(input=user_question, context=context)
    
    full_response = f"{response}\n"

    return jsonify({
        'response': full_response,
        'context_used': context,
    })

@main.teardown_app_request
def close_neo4j_driver(exception=None):
    driver.close()
# import os
# from dotenv import load_dotenv
# import asyncio
# from neo4j import GraphDatabase
# from concurrent.futures import ThreadPoolExecutor
# import logging
# from .entity_check import extract_entities

# load_dotenv()
# logger = logging.getLogger(__name__)

# class Neo4jOperations:
#     def __init__(self):
#         self.URI = os.getenv("NEO4J_URI")
#         self.USER = os.getenv("NEO4J_USER")
#         self.PASSWORD = os.getenv("NEO4J_PASSWORD")
#         self.driver = GraphDatabase.driver(self.URI, auth=(self.USER, self.PASSWORD))
#         self.thread_pool = ThreadPoolExecutor(max_workers=10)
#         self.query_cache = {}

#     def construct_cypher_query(self, entities, question):
#         ranked_entities = rank_entities(entities)
#         query_parts = ["CALL db.index.fulltext.queryNodes('startupIndex', $search_term) YIELD node as s, score"]
#         where_clauses = []
#         params = {'search_term': question}  

#         for category, entity in ranked_entities[:3]: 
#             if category == 'CATEGORY':
#                 where_clauses.append(f"(s.name =~ $regex_{len(where_clauses)} OR s.category =~ $regex_{len(where_clauses)})")
#             elif category == 'LOCATION':
#                 where_clauses.append(f"s.location =~ $regex_{len(where_clauses)}")
#             elif category == 'STATUS':
#                 where_clauses.append(f"s.status =~ $regex_{len(where_clauses)}")
#             params[f'regex_{len(where_clauses)-1}'] = f"(?i).*{entity}.*"

#         if where_clauses:
#             query_parts.append("WHERE " + " OR ".join(where_clauses))

#         query_parts.append("RETURN s.name AS name, s.location AS location, s.category AS category, s.status AS status, score")
#         query_parts.append("ORDER BY score DESC")
#         query_parts.append("LIMIT 5")

#         return " ".join(query_parts), params

#     async def query_neo4j_async(self, question):
#         entities = extract_entities(question)
#         results = {}
        
#         for entity_type, entity_values in entities.items():
#             results[entity_type] = []
#             for entity_value in entity_values:
#                 cache_key = f"{entity_type}:{entity_value}"
#                 if cache_key in self.query_cache:
#                     results[entity_type].extend(self.query_cache[cache_key])
#                     continue

#                 query = """
#                 MATCH (s:Startup)
#                 WHERE toLower(s.name) CONTAINS toLower($entity_value)
#                 OR (s)-[:LOCATED_IN]->(:Location {name: $entity_value})
#                 OR (s)-[:IN_CATEGORY]->(:Category {name: $entity_value})
#                 OR (s)-[:HAS_STATUS]->(:Status {name: $entity_value})
#                 WITH s
#                 OPTIONAL MATCH (s)-[:LOCATED_IN]->(l:Location)
#                 OPTIONAL MATCH (s)-[:IN_CATEGORY]->(c:Category)
#                 OPTIONAL MATCH (s)-[:HAS_STATUS]->(st:Status)
#                 RETURN s.name AS name, l.name AS location, c.name AS category, st.name AS status
#                 ORDER BY s.relevance DESC
#                 LIMIT 5
#                 """
                
#                 loop = asyncio.get_event_loop()
#                 try:
#                     with self.driver.session() as session:
#                         result = await loop.run_in_executor(
#                             self.thread_pool,
#                             lambda: session.run(query, entity_value=entity_value).data()
#                         )
#                     self.query_cache[cache_key] = result
#                     results[entity_type].extend(result)
#                 except Exception as e:
#                     logger.error(f"Error querying Neo4j: {str(e)}")
        
#         return results

#     def close(self):
#         self.driver.close()
    
#     @classmethod
#     def create(cls):
#         return cls()

import os
from dotenv import load_dotenv
import asyncio
from neo4j import GraphDatabase
from concurrent.futures import ThreadPoolExecutor
import logging
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
logger = logging.getLogger(__name__)

class Neo4jOperations:
    def __init__(self):
        self.URI = os.getenv("NEO4J_URI")
        self.USER = os.getenv("NEO4J_USER")
        self.PASSWORD = os.getenv("NEO4J_PASSWORD")
        self.driver = GraphDatabase.driver(self.URI, auth=(self.USER, self.PASSWORD))
        self.thread_pool = ThreadPoolExecutor(max_workers=10)
        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name='llama3-8b-8192',
            temperature=0,
            max_tokens=300
        )

    async def generate_cypher_query(self, question):
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert in translating natural language questions about startups into Neo4j Cypher queries.
            The graph has the following structure:
            (Startup)-[:LOCATED_IN]->(Location)
            (Startup)-[:IN_CATEGORY]->(Category)
            (Startup)-[:HAS_STATUS]->(Status)
            
            Startup properties: name, relevance
            Location properties: name
            Category properties: name
            Status properties: name

            Always use MATCH clauses and avoid using CALL db.index.fulltext.queryNodes().
            Limit results to 5 unless explicitly asked for a different number or for a count.
            Only return properties that exist in the graph model.
            Do not invent or assume any information not explicitly defined in the graph model.
            Your task is ONLY to generate a valid Cypher query, not to provide any other information or explanation.
            """),
            ("human", "Translate this question into a Cypher query: {question}")
        ])

        chain = LLMChain(llm=self.llm, prompt=prompt, output_parser=StrOutputParser())
        
        cypher_query = await chain.arun(question=question)
        return cypher_query.strip()

    async def execute_cypher_query(self, query):
        loop = asyncio.get_event_loop()
        try:
            with self.driver.session() as session:
                result = await loop.run_in_executor(
                    self.thread_pool,
                    lambda: session.run(query).data()
                )
            return result
        except Exception as e:
            logger.error(f"Error executing Cypher query: {str(e)}")
            return []

    async def query_neo4j_async(self, question):
        cypher_query = await self.generate_cypher_query(question)
        logger.info(f"Generated Cypher query: {cypher_query}")
        return await self.execute_cypher_query(cypher_query)

    def close(self):
        self.driver.close()
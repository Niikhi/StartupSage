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
from neo4j import AsyncGraphDatabase
import logging
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from functools import lru_cache
import traceback

load_dotenv()
logger = logging.getLogger(__name__)

class Neo4jOperations:
    def __init__(self):
        self.URI = os.getenv("NEO4J_URI")
        self.USER = os.getenv("NEO4J_USER")
        self.PASSWORD = os.getenv("NEO4J_PASSWORD")
        self.llm = self.init_llm()
        self.driver = None
        self.is_connected = False
        logger.info(f"Initializing Neo4jOperations with URI: {self.URI}, USER: {self.USER}")

    async def connect(self):
        try:
            logger.info("Attempting to connect to Neo4j database...")
            if not self.URI or not self.USER or not self.PASSWORD:
                raise ValueError("Neo4j connection details are missing")

            self.driver = AsyncGraphDatabase.driver(self.URI, auth=(self.USER, self.PASSWORD))
            
            # Test the connection asynchronously
            async def test_connection():
                async with self.driver.session() as session:
                    result = await session.run("RETURN 1 as num")
                    record = await result.single()
                    return record and record["num"] == 1

            is_connected = await test_connection()
            if (is_connected):
                logger.info("Successfully connected to Neo4j database")
                self.is_connected = True
            else:
                logger.error("Failed to connect to Neo4j database")
                self.is_connected = False
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {str(e)}")
            logger.error(traceback.format_exc())
            self.is_connected = False

    @staticmethod
    def init_llm():
        return ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name='llama3-8b-8192',
            temperature=0,
            max_tokens=300
        )


    @lru_cache(maxsize=100)
    async def generate_cypher_query(self, question):
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert in translating natural language questions about startups into Neo4j Cypher queries.
            The graph has the following structure:
            (Startup)-[:LOCATED_IN]->(Location)
            (Startup)-[:IN_CATEGORY]->(Category)
            (Startup)-[:HAS_STATUS]->(Status)
            
            Startup properties: name
            Location properties: name
            Category properties: name
            Status properties: name

            Guidelines for generating Cypher queries:

            1. For questions about startups in a specific location:
            MATCH (s:Startup)-[:LOCATED_IN]->(l:Location)
            WHERE toLower(l.name) CONTAINS toLower('location name')
            RETURN s.name, l.name

            2. For questions about startups in a specific category:
            MATCH (s:Startup)-[:IN_CATEGORY]->(c:Category)
            WHERE toLower(c.name) CONTAINS toLower('category name')
            RETURN s.name, c.name

            3. For questions about startups with a specific status:
            MATCH (s:Startup)-[:HAS_STATUS]->(st:Status)
            WHERE toLower(st.name) CONTAINS toLower('status name')
            RETURN s.name, st.name

            4. For questions combining multiple aspects (e.g., category and location):
            MATCH (s:Startup)-[:IN_CATEGORY]->(c:Category), (s)-[:LOCATED_IN]->(l:Location)
            WHERE toLower(c.name) CONTAINS toLower('category name')
            AND toLower(l.name) CONTAINS toLower('location name')
            RETURN s.name, c.name, l.name

            5. For general questions about startup counts:
            MATCH (s:Startup)
            RETURN count(s) as startup_count

            6. For questions about top startups (assuming there's a 'ranking' property):
            MATCH (s:Startup)
            RETURN s.name, s.ranking
            ORDER BY s.ranking DESC
            LIMIT 10

            Always limit results to 5 unless specified otherwise.
            Use CONTAINS for flexible matching of names.
            If the question doesn't fit any of these patterns, return 'INVALID QUERY'.
            Your task is ONLY to generate a valid Cypher query, not to provide any other information or explanation.
            """),
            ("human", "Translate this question into a Cypher query: {question}")
        ])

        chain = LLMChain(llm=self.llm, prompt=prompt, output_parser=StrOutputParser())
        
        response = await chain.ainvoke({"question": question})
        return response["text"].strip()

    async def execute_cypher_query(self, query, params=None):
        try:
            async with self.driver.session() as session:
                result = await session.run(query, params)
                return await result.data()
        except Exception as e:
            logger.error(f"Error executing Cypher query: {str(e)}")
            return []

    async def fallback_query(self):
        query = """
        MATCH (s:Startup)-[:IN_CATEGORY]->(c:Category)
        WHERE toLower(c.name) CONTAINS toLower('social') OR toLower('network')
        RETURN s.name as name, c.name as category, s.relevance as relevance
        ORDER BY s.relevance DESC
        LIMIT 10
        """
        return await self.execute_cypher_query(query)

    async def query_neo4j_async(self, question):
        if not self.is_connected:
            logger.error("Not connected to Neo4j database")
            return None

        initial_query = await self.generate_cypher_query(question)
        logger.info(f"Generated initial Cypher query: {initial_query}")
        
        if initial_query == 'INVALID QUERY':
            return []

        try:
            initial_results = await self.execute_cypher_query(initial_query)
            
            if initial_results:
                startup_names = [result['s.name'] for result in initial_results[:5]]
                detailed_query = """
                MATCH (s:Startup)
                WHERE s.name IN $names
                OPTIONAL MATCH (s)-[:LOCATED_IN]->(l:Location)
                OPTIONAL MATCH (s)-[:IN_CATEGORY]->(c:Category)
                OPTIONAL MATCH (s)-[:HAS_STATUS]->(st:Status)
                RETURN s.name as name, l.name as location, c.name as category, st.name as status
                """
                detailed_results = await self.execute_cypher_query(detailed_query, {"names": startup_names})
                return detailed_results
            return []
        except Exception as e:
            logger.error(f"Error querying Neo4j: {str(e)}", exc_info=True)
            return []

    async def close(self):
        if self.driver:
            try:
                await self.driver.close()
            except Exception as e:
                logger.error(f"Error closing Neo4j driver: {str(e)}")

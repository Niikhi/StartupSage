import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
import logging

load_dotenv()

URI = os.getenv("NEO4J_URI")
USER = os.getenv("NEO4J_USER")
PASSWORD = os.getenv("NEO4J_PASSWORD") 

driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_fulltext_index():
    with driver.session() as session:
        try:
            # Try the newer syntax first
            result = session.run("SHOW INDEXES WHERE name = 'startupIndex'")
            if not list(result):
                logger.warning("Full-text index 'startupIndex' does not exist. Creating it now.")
                create_fulltext_index()
            else:
                logger.info("Full-text index 'startupIndex' already exists.")
        except Exception:
            try:
                # Fall back to older syntax if the above fails
                result = session.run("CALL db.indexes() YIELD name, type WHERE name = 'startupIndex'")
                if not list(result):
                    logger.warning("Full-text index 'startupIndex' does not exist. Creating it now.")
                    create_fulltext_index()
                else:
                    logger.info("Full-text index 'startupIndex' already exists.")
            except Exception as e:
                logger.error(f"Error checking for full-text index: {str(e)}")
                logger.warning("Unable to check for 'startupIndex'. Please ensure it exists manually.")



def create_fulltext_index():
    with driver.session() as session:
        result = session.run("CALL db.indexes() YIELD name, labelsOrTypes, properties WHERE name = 'startupIndex' RETURN count(*) as count")
        if result.single()['count'] == 0:
            print("Creating full-text index...")
            session.run("""
            CALL db.index.fulltext.createNodeIndex("startupIndex", ["Startup"], ["name", "category", "location", "status"])
            """)
            print("Full-text index created successfully.")
        else:
            print("Full-text index already exists.")

if __name__ == "__main__":
    create_fulltext_index()
    driver.close()
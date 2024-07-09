import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
import sqlite3
import pandas as pd
from nltk.corpus import wordnet
import nltk

nltk.download('wordnet', quiet=True)

load_dotenv()

URI = os.getenv("NEO4J_URI")
USER = os.getenv("NEO4J_USER")
PASSWORD = os.getenv("NEO4J_PASSWORD")


def load_data_from_sqlite():
    conn = sqlite3.connect('startup_data.db')
    query = "SELECT id, name, location, category, status FROM startups"
    data = pd.read_sql_query(query, conn)
    conn.close()
    return data

def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().lower().replace('_', ' '))
    return list(synonyms)

def create_neo4j_graph(data):
    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
    
    try:
        with driver.session() as session:
            # Create constraints and indexes (code remains the same)
            constraints_and_indexes = [
                "CREATE CONSTRAINT startup_name IF NOT EXISTS FOR (s:Startup) REQUIRE s.name IS UNIQUE",
                "CREATE CONSTRAINT location_name IF NOT EXISTS FOR (l:Location) REQUIRE l.name IS UNIQUE",
                "CREATE CONSTRAINT category_name IF NOT EXISTS FOR (c:Category) REQUIRE c.name IS UNIQUE",
                "CREATE CONSTRAINT status_name IF NOT EXISTS FOR (st:Status) REQUIRE st.name IS UNIQUE",
                "CREATE INDEX startup_id IF NOT EXISTS FOR (s:Startup) ON (s.id)",
                "CREATE INDEX category_synonym IF NOT EXISTS FOR (c:Category) ON (c.synonym)"
            ]
            for query in constraints_and_indexes:
                try:
                    session.run(query)
                except Exception as e:
                    print(f"Constraint/Index creation failed (it might already exist): {e}")

            # Process startups
            for _, row in data.iterrows():
                # Check if startup exists
                result = session.run("""
                    MATCH (s:Startup {name: $name})
                    RETURN s
                """, {'name': row['name']})
                
                if not result.single():
                    # Create Startup
                    session.run("""
                        MERGE (s:Startup {name: $name})
                        ON CREATE SET s.id = $id
                    """, {'name': row['name'], 'id': row['id']})

                    # Create and connect Location
                    session.run("""
                        MERGE (l:Location {name: $location})
                        WITH l
                        MATCH (s:Startup {name: $name})
                        MERGE (s)-[:LOCATED_IN]->(l)
                    """, {'name': row['name'], 'location': row['location']})

                    # Create and connect Status
                    session.run("""
                        MERGE (st:Status {name: $status})
                        WITH st
                        MATCH (s:Startup {name: $name})
                        MERGE (s)-[:HAS_STATUS]->(st)
                    """, {'name': row['name'], 'status': row['status']})

                    # Create and connect Category (including synonyms)
                    category_synonyms = get_synonyms(row['category'])
                    category_synonyms.append(row['category'])  # Include the original category
                    for synonym in set(category_synonyms):  # Use set to remove duplicates
                        session.run("""
                            MERGE (c:Category {name: $category})
                            ON CREATE SET c.synonym = $synonym
                            WITH c
                            MATCH (s:Startup {name: $name})
                            MERGE (s)-[:IN_CATEGORY]->(c)
                        """, {'name': row['name'], 'category': row['category'], 'synonym': synonym})

                    print(f"Processed startup: {row['name']}")
                else:
                    print(f"Skipped existing startup: {row['name']}")

            print("Neo4j graph update completed.")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        driver.close()

if __name__ == "__main__":
    data = load_data_from_sqlite()
    create_neo4j_graph(data)


#     To further improve this structure:

# Implementing a caching mechanism for synonyms to reduce repeated calls to WordNet.
# Using Neo4j's APOC library for even more efficient batch imports if you're dealing with very large datasets.
# Implementing error logging and transaction management for better error handling and data consistency.
import sqlite3
import json

def extract_keywords_from_db(db_path='startup_data.db'):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    entity_keywords = {
        'CATEGORY': set(),
        'STATUS': set(),
        'LOCATION': set()
    }

    # Extract categories
    cursor.execute("SELECT DISTINCT category FROM startups")
    entity_keywords['CATEGORY'] = set(row[0] for row in cursor.fetchall() if row[0])

    # Extract statuses
    cursor.execute("SELECT DISTINCT status FROM startups")
    entity_keywords['STATUS'] = set(row[0] for row in cursor.fetchall() if row[0])

    # Extract locations
    cursor.execute("SELECT DISTINCT location FROM startups")
    entity_keywords['LOCATION'] = set(row[0] for row in cursor.fetchall() if row[0])

    conn.close()

    # Convert sets to lists for JSON serialization
    return {k: list(v) for k, v in entity_keywords.items()}

# Generate and save the JSON file
keywords = extract_keywords_from_db()
with open('entity_keywords.json', 'w') as f:
    json.dump(keywords, f, indent=4)

print("entity_keywords.json has been generated.")
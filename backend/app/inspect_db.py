import sqlite3
import pandas as pd

conn = sqlite3.connect('startup_data.db')
query = "SELECT name, location, category, status FROM startups"

try:
    data = pd.read_sql_query(query, conn)
    print("Table 'startups' exists and contains the following data:")
    print(data.head())
except Exception as e:
    print(f"Error: {e}")
finally:
    conn.close()

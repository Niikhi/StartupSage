# import sqlite3
# import pandas as pd
# import json
# from langchain_community.vectorstores import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings  
# from langchain_community.document_loaders import DataFrameLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter  


# def load_data_from_db():
#     conn = sqlite3.connect('startup_data.db')
#     query = "SELECT id, name, location, category, status FROM startups"
#     data = pd.read_sql_query(query, conn)
#     conn.close()


#     data['combined_text'] = data[['name', 'location', 'category', 'status']].agg(' '.join, axis=1)

#     json_data = data.to_dict(orient='records')
#     with open('startups_data.json', 'w') as json_file:
#         json.dump(json_data, json_file, indent=4)

#     print("Data successfully saved to startups_data.json.")
#     return data

# def create_chromadb_index(data):
#     embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#     df_loader = DataFrameLoader(data, page_content_column='combined_text')
#     df_documents = df_loader.load()

#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=10)
#     texts = text_splitter.split_documents(df_documents)

#     max_batch_size = 166  
#     for i in range(0, len(texts), max_batch_size):
#         batch_texts = texts[i:i+max_batch_size]
#         chromadb_index = Chroma.from_documents(
#             batch_texts, embedding_function, persist_directory='./chroma_db'
#         )

#     print("Embeddings successfully stored in ChromaDB.")

# if __name__ == '__main__':
#     data = load_data_from_db()
#     create_chromadb_index(data)

# ------------------- imp using docs ------------------------------------ #
import sqlite3
import pandas as pd
import json
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings  
from langchain_community.document_loaders import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter  
from tqdm import tqdm

def load_data_from_db():
    conn = sqlite3.connect('startup_data.db')
    query = "SELECT id, name, location, category, status FROM startups"
    data = pd.read_sql_query(query, conn)
    conn.close()

    documents = []
    for _, row in data.iterrows():
        doc = f"Startup: {row['name']}\nLocation: {row['location']}\nCategory: {row['category']}\nStatus: {row['status']}"
        documents.append(doc)

    return documents

def create_chromadb_index(documents):
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.create_documents(documents)

    max_batch_size = 166  # Maximum batch size allowed by Chroma
    chromadb_index = None

    for i in tqdm(range(0, len(texts), max_batch_size), desc="Creating Chroma index"):
        batch_texts = texts[i:i+max_batch_size]
        
        if chromadb_index is None:
            chromadb_index = Chroma.from_documents(
                batch_texts, embedding_function, persist_directory='./chroma_db'
            )
        else:
            chromadb_index.add_documents(batch_texts)

if __name__ == '__main__':
    documents = load_data_from_db()
    create_chromadb_index(documents)


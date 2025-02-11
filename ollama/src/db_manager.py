from pathlib import Path
import sys
root_path = Path(__file__).resolve().parents[4]
sys.path.append(str(root_path))

from .data import load_pdf, split_documents
from .vector_db import ChromaDB
from .embedding_fn import CustomOllamaEmbeddings
from .db_query import query_database
from .llm_response import query_rag
import json
from .config import Config
import os

def add_document_to_db(pdf_path: str, vector_db):
    documents_data = load_pdf(pdf_path)
    split_documents_data = split_documents(documents_data)
    vector_db.add_chunks(split_documents_data)
    # print(f"Data added to collection '{collection.name}'.")

# Initialize ChromaDBManager instance
class ChromaDBManager:
    def __init__(self):
        self.vector_db = ChromaDB()
        self.collection = None
        self.model = query_rag(Config.MODEL)


    def get_list_collections(self):
        return self.vector_db.get_list_collections()

    def get_collection(self, collection_name: str, embedding_function):
        return self.vector_db.get_collection(collection_name, CustomOllamaEmbeddings())

    def initialize_database(self, pdf_path: str, collection_name: str, debug: bool = False):
        self.collection = self.vector_db.add_collection(collection_name, CustomOllamaEmbeddings())
        documents_data = load_pdf(pdf_path)
        if debug:
            print("Loaded Documents Data:", documents_data)

        split_documents_data = split_documents(documents_data)
        if debug:
            print("Split Documents Data:", split_documents_data)

        self.vector_db.add_chunks(split_documents_data)

        print(f"Collection '{collection_name}' created and data added.")

    def initialize_database_from_folder(self, folder_path: str, collection_name: str, debug: bool = False):
        """ Loads a folder containing PDF files and adds them to the database """


        pdf_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.pdf')]
        if not pdf_files:
            print("No PDF files found in the folder.")
            return
        print(f"Found {len(pdf_files)} PDF files in the folder.")

        # Try to load a collection with the same name if it exists
        if collection_name in self.get_list_collections():
            print(f"Collection '{collection_name}' already exists.")
            return
        
        self.collection = self.vector_db.add_collection(collection_name, CustomOllamaEmbeddings())

        for pdf_file in pdf_files:
            add_document_to_db(pdf_file, self.vector_db)

        print(f"Collection '{collection_name}' created and data added.")

       
    def delete_collection(self, collection_name: str):
        self.collection = self.vector_db.get_collection(collection_name, CustomOllamaEmbeddings())
        if self.collection:
            self.vector_db.chroma_client.delete_collection(collection_name)
            print(f"Collection '{collection_name}' deleted.")
        else:
            print(f"Collection '{collection_name}' not found.")


    def query_database(self, query_text: str, collection_name: str, debug: bool = False):
        collection = self.get_collection(collection_name, CustomOllamaEmbeddings())
        query_context, prompt = query_database(collection, query_text)

        if debug:
            print("Query Context:", query_context)
            print("Prompt:", prompt)

        rag_response = self.model.invoke(query_context, prompt)
        print("RAG Response:", rag_response)

        return rag_response
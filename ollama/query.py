from src.config import Config
from src.db_manager import ChromaDBManager
from src.embedding_fn import CustomOllamaEmbeddings
import argparse

def get_collection(collection_name):
    db_manager = ChromaDBManager()
    collection = db_manager.get_collection(collection_name, CustomOllamaEmbeddings())
    if not collection:
        print(f"Collection '{collection_name}' not found.")
        exit()
    return collection

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', '-n', required=True, help='Name of the collection. To be used for querying.')
    parser.add_argument('--query', '-q', help='Query text')
    parser.add_argument('--file_in', '-i', help='Path to input txt file containing query text')
    parser.add_argument('--file_out', '-o', help='Path to output txt file containing query results')
    args = parser.parse_args()

    db_manager = ChromaDBManager()
    collection_name = args.name
    collection = db_manager.get_collection(collection_name, CustomOllamaEmbeddings())

    # Updated logic for reading query
    if args.file_in:
        with open(args.file_in, 'r') as f:
            query = f.read().strip()
    else:
        query = args.query

    # Execute the query
    response = db_manager.query_database(query, collection_name, debug=True)

    if args.file_out:
        with open(args.file_out, 'w') as f:
            f.write(response)
    else:
        print(response)
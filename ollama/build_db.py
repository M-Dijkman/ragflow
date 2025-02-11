from src.config import Config
from src.db_manager import ChromaDBManager
from src.embedding_fn import CustomOllamaEmbeddings 
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-p', required=True, help='Path to the folder containing pdf files')
    parser.add_argument('--name', '-n', required=True, help='Name of the collection. To be used for querying.')

    args = parser.parse_args()
    db_manager = ChromaDBManager()
    db_manager.initialize_database_from_folder(args.path, args.name, debug=True)    
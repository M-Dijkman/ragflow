from src.config import Config
from src.db_manager import ChromaDBManager
from src.embedding_fn import CustomOllamaEmbeddings 
from src.rag_test import generate_questions_answers



def main():
    import argparse
    parser = argparse.ArgumentParser(description='This is the RAG model cli interface')
    parser.add_argument('--debug', action='store_true', help='Print debug information')
    args = parser.parse_args()

    db_manager = ChromaDBManager()

    while True:
        print("\nOptions:")
        print("1. Load PDF and create collection")
        print("2. Query the database")
        print("3. Delete a specific collection")
        print("4. Test document retrieval")
        print("5. Exit")

        choice = input("Enter the number of your choice: ")

        if choice == '1':
            pdf_path = input("Enter the path to the PDF file: ")
            collection_name = input("Enter the name of the collection: ")
            db_manager.initialize_database(pdf_path, collection_name, args.debug)

        elif choice == '2':
            print("Available collections:")
            collections_list = db_manager.get_list_collections()
            for collection in collections_list:
                print(collection.name)
                # pass
            collection_name = input("Enter the name of the collection to query: ")

            if not collection_name:
                print("Collection name cannot be empty.")
                continue
            if collection_name not in [collection.name for collection in collections_list]:
                print(f"Collection '{collection_name}' not found.")
                continue
        
            collection = db_manager.get_collection(collection_name, CustomOllamaEmbeddings())
            if not collection:
                print(f"Collection '{collection_name}' not found.")
                continue

            # db_manager.interactive_query(collection, args.debug)
            query_text = input("Enter your query: ")
            db_manager.query_database(query_text, collection_name, args.debug)


        elif choice == '3':
            collection_name = input("Enter the name of the collection to delete: ")
            db_manager.delete_collection(collection_name)

        elif choice == '4':
            print("work in progress")
            generate_questions_answers()

        elif choice == '5':
            print("Exiting...")
            break

        else:
            print("Invalid choice,pick the right option.")





if __name__ == '__main__':
    main()
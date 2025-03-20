from langchain.prompts import ChatPromptTemplate
from .config import Config, PREPROMPT

def query_database(collection, query_text: str):

    results = collection.query(
            query_texts=[query_text], n_results=int(Config.RAG_TOP_K)
            , include=["documents", "metadatas"]
        )

    context_level = results["documents"][0]

    prompt_template = ChatPromptTemplate.from_template(PREPROMPT)
    prompt = prompt_template.format(context=context_level, question=query_text)

    return results, prompt
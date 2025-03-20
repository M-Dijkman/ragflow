# from langchain_community.llms.ollama import Ollama
from langchain_ollama import OllamaLLM as Ollama


class query_rag:
    def __init__(self, model):
        self.model = Ollama(model=model)

    def invoke(self, query_results, prompt):
        response = self.model.invoke(prompt)

        sources = [doc.get('chunk_id') for doc in query_results['metadatas'][0]]
    
        formated_response = f"Response: {response}\n\nSources: {sources}"

        return formated_response
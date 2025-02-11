class Config:
    CHROMA_PATH = 'database/vector_db'
    CHUNK_SIZE = 2000
    CHUNK_OVERLAP = CHUNK_SIZE * 0.05
    RAG_TOP_K = 5
    DATA_CHAT_PATH = 'database/chat'
    MODEL = 'llama3'


PREPROMPT = """
Please answer the question and use only information found in the following context: 

{context}

----------------------------------------

Answer the question in word or number based on the above context: {question}, and if you are not sure
about the answer, give a relevant answer based on the context and give certainty score of 0.5


"""
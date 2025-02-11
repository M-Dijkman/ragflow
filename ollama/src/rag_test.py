import os
import json
from src.db_manager import ChromaDBManager as pd    
from src.config import Config
from src.db_manager import ChromaDBManager
from src.embedding_fn import CustomOllamaEmbeddings 
from langchain.chat_models import ChatOllama
from langchain.chains import LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader

def generate_questions_answers():
    # Define the prompt template
    PREPROMT = """
        You are an expert on generating a question-and-answer dataset based on a given context. 
        You are given a context. Your task is to generate a question and answer based on the context. 
        The generated question should be able to answer by leveraging the given context. 
        The generated question-and-answer pair must be grammatically and semantically correct. 
        Your response must be in a JSON format with two keys: question, answer.
        ...
    """
    
    # Initialize the LLM from Ollama
    llm = ChatOllama(model="llama3.3", temperature=0.3)

    # Create the LLM chain
    chain = LLMChain(
        prompt=PREPROMT,
        llm=llm,
    )

    # Load and process the PDF
    file_loader = PyMuPDFLoader("./database/ragtest.pdf")
    text_splitter = CharacterTextSplitter(chunk_size=15, chunk_overlap=10)
    chunks = text_splitter.split_documents(file_loader.load())

    # Generate questions and answers
    questions, answers = [], []
    for chunk in chunks:
        response = chain.invoke({"context": chunk.page_content})
        obj = json.loads(response["text"])
        questions.append(obj["question"])
        answers.append(obj["answer"])

    # Save to CSV
    df = pd.DataFrame({"question": questions, "answer": answers})
    df.to_csv("./database/AlpaFold_qa_llama3.csv", index=False)
    print("Questions and answers have been saved to './database/AlpaFold_qa_llama3.csv'")
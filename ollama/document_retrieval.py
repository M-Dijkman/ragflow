import pandas as pd
from tqdm import tqdm
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOllama
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings


# Load the questions and answers from the previously created CSV
csv_file_path = "./database/AlphaFold_qa.csv"
df = pd.read_csv(csv_file_path)

chunks = 2
# Initialize the FAISS vector store with HuggingFace embeddings
# Replace with actual document chunking code if necessary
vector_store = FAISS.from_documents(chunks, HuggingFaceEmbeddings())
retriever = vector_store.as_retriever()

# Define a function to test only on `llama3`
def test_llama3_retrieval_qa(model="llama3"):
    # Create the retrieval QA chain
    chain = RetrievalQA.from_llm(
        llm=ChatOllama(model=model),
        retriever=retriever,
    )
    
    # Initialize predictions
    predictions = []
    print(f"Running retrieval QA with {model}...")
    
    # Iterate through the questions in the DataFrame
    for _, row in tqdm(df.iterrows(), total=len(df)):
        resp = chain.invoke({
            "query": row["question"]
        })
        predictions.append(resp["result"])
    
    # Add results to the DataFrame
    df[f"{model}_result"] = predictions

    # Save results to a new CSV file
    output_file_path = "./database/AlphaFold_qa_retrieval_prediction.csv"
    df.to_csv(output_file_path, index=False)
    print(f"Results saved to {output_file_path}")

# Run the test for `llama3`
test_llama3_retrieval_qa()
from langchain.schema.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .config import Config
    
from typing import List, Dict, Any
from langchain_community.document_loaders import PyMuPDFLoader


def load_pdf(file_path: str) -> Dict[str, Any]:
    """
    Load the pdf file and return the text context
    """
    loader = PyMuPDFLoader(file_path)
    return loader.load()

def split_documents(documents: list[Document]):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size= Config.CHUNK_SIZE,
        chunk_overlap= Config.CHUNK_OVERLAP,
        length_function= len,
        is_separator_regex = False,
    )
    return splitter.split_documents(documents)
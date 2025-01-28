from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from io import BytesIO
from langchain_community.document_loaders import UnstructuredURLLoader

def load_data(data_path):
    """
    Load PDF documents from a directory.
    
    Args:
        data_path (str): Path to the directory containing PDF files.

    Returns:
        List[Document]: List of loaded documents.
    """
    loader = DirectoryLoader(data_path, glob="*.pdf", loader_cls=PyPDFLoader)
    data = loader.load()
    return data

def load_data_from_uploaded_pdf(file):
    """
    Load a single uploaded PDF file.

    Args:
        file: File object of the uploaded PDF.

    Returns:
        List[Document]: List of loaded documents.
    """
    loader = PyPDFLoader(file)
    data = loader.load()
    return data

def load_data_from_url(url):
    """
    Load data from a URL.

    Args:
        url (str): URL to scrape data from.

    Returns:
        List[Document]: List of loaded documents.
    """
    urls = [f"{url}"]
    loader = UnstructuredURLLoader(urls)
    data = loader.load()
    return data

def text_split(data):
    # Assuming docs is a list of text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    doc_chunks = text_splitter.split_documents(data)
    
    # Print the result to check if the chunks are generated
    if len(doc_chunks) == 0:
        print("No chunks created. Docs length:", len(data))
    return doc_chunks



def download_huggingface_embedding():
    """
    Download Hugging Face embeddings model.

    Returns:
        HuggingFaceEmbeddings: Hugging Face embeddings object.
    """
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings

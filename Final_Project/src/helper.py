from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

def load_data(data_path):
    loader = DirectoryLoader(data_path, glob="*.pdf", loader_cls=PyPDFLoader)
    data = loader.load()
    return data

def load_data_from_uploaded_pdf(file):
    loader = PyPDFLoader(file)
    data = loader.load()
    return data

def load_data_from_url(url):
    urls = [f"{url}"]
    loader = UnstructuredURLLoader(urls)
    data = loader.load()
    return data

def text_split(data):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    doc_chunks = splitter.split_documents(data)
    
    if len(doc_chunks) == 0:
        print("No chunks created. Docs length:", len(data))
    return doc_chunks

def download_huggingface_embedding():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings


import streamlit as st
import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from src.helper import download_huggingface_embedding, load_data_from_uploaded_pdf,load_data_from_url, text_split

def main():
    # Basic configurations
    PINECONE_INDEX_NAME = "medical-chatbot"
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    # Load HuggingFace embeddings
    embeddings = download_huggingface_embedding()
    
    # Check if embeddings are empty or invalid
    if not embeddings:
        st.error("Embeddings were not generated properly.")
        return

    load_dotenv()

    # Configure Streamlit page settings
    st.set_page_config(
        page_title="Healthcare-bot",
        page_icon="🤖",
        layout="centered"
    )

    st.title("      🏥  Healthcare - Bot  🩺 ")
    st.write("Choose how you want to provide medical data for the chatbot.")

    # Sidebar inputs
    uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type="pdf")
    url = st.sidebar.text_input("Enter a URL (optional)")
    use_default = st.sidebar.checkbox("Use default data", value=True)

    # Placeholder for processing user input
    if uploaded_file:
        print(uploaded_file)
        st.success(f"PDF uploaded successfully! Processing '{uploaded_file.name}'...")
        with open("uploaded_file.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        docs = load_data_from_uploaded_pdf("uploaded_file.pdf")
        doc_chucks = text_split(docs)
        
        # Check if document chunks are empty
        if not doc_chucks:
            st.error("No document chunks generated.")
            return
        
        docsearch = Chroma.from_documents(
            documents=doc_chucks,
            embedding=embeddings,
            collection_name="PDF_database",
            persist_directory="./chroma_db_PDF"
        )

    elif url:
        st.success("URL provided: {}".format(url))
        docs = load_data_from_url(url=url)
        doc_chucks = text_split(docs)

        # Check if document chunks are empty
        if not doc_chucks:
            st.error("No document chunks generated.")
            return
        
        docsearch = Chroma.from_documents(
            documents=doc_chucks,
            embedding=embeddings,
            collection_name="URL_database",
            persist_directory="./chroma_db_url"
        )
        docsearch.add_documents(doc_chucks)
        st.success("Index loaded successfully!")

    elif use_default:
        st.success("Using default GALA ENCYCLOPEDIA OF MEDICINE data!")
        try:
            docsearch = PineconeVectorStore.from_existing_index(PINECONE_INDEX_NAME, embeddings)
            st.success("Index loaded successfully!")
        except Exception as e:
            st.error(f"Error loading index: {e}")
    else:
        st.info("Please upload a file, enter a URL, or select default data to proceed.")
        st.stop()

    # Define prompt template
    prompt_template = """
    Use ONLY the given information context to generate an appropriate answer for the user's question.
    If the answer is not present in the context, respond with "I don't know the answer based on the provided information."

    Context: {context}
    Question: {question}
    Only return the appropriate answer based strictly on the given context.
    Helpful answer:
    """

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    # Initialize RetrievalQA chain with a specific output key for 'result'
    chain_type_kwargs = {"prompt": PROMPT}

    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model="mixtral-8x7b-32768",
        temperature=0.5,
        max_tokens=1000,
        timeout=60
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=docsearch.as_retriever(search_kwargs={'k': 4}),
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs,
    )

    # Initialize session state for chat history
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # Chat input and processing
    input = st.chat_input("Say something")
    if input:
        result = qa.invoke(input)
        print("Response: ",result["result"])
        response = result["result"]
        st.session_state["chat_history"].append((input, response))

    # Display chat history
    for question, answer in st.session_state["chat_history"]:
        st.write(f" 🧑‍💼 :  {question}")  
        st.write(f" 💉 :  {answer}") 


if __name__ == "__main__":
    main()      


# RAG-based AI Agent for Healthcare

## Project Overview

This project aims to build a healthcare chatbot using Retrieval-Augmented Generation (RAG) techniques. The chatbot utilizes a combination of LangChain, Groq, ChromaDB, and Streamlit to provide intelligent and context-aware answers based on the given medical data. Users can interact with the bot by uploading PDFs, entering URLs, or using default pre-uploaded data.

### Technologies Used:
- **LangChain**: A framework to work with language models and handle tasks like document loading, embedding, and text splitting.
- **Groq**: A high-performance model for fast inference.
- **ChromaDB**: A vector database used for storing and searching document embeddings.
- **Streamlit**: A web framework for creating interactive web applications.
- **Python**: The programming language used to implement the project.

## Features
- **Upload PDF**: Users can upload PDF files containing medical data to be processed by the chatbot.
- **Enter URL**: Users can provide a URL, and the chatbot will scrape and process the content to answer questions.
- **Use Default Data**: The chatbot can use a pre-uploaded set of medical data from GALA ENCYCLOPEDIA OF MEDICINE for answering queries.
- **Medical-related Q&A**: Users can ask medical questions, and the bot will provide context-aware answers using the loaded documents or default data.

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/your-username/rag_based_ai_agent_for_healthcare.git
cd rag_based_ai_agent_for_healthcare
```

### 2. Install dependencies
Make sure you have Python 3.12+ installed. You can use `pip` to install the necessary packages:
```bash
pip install -r requirements.txt
```

### 3. Set up environment variables
Create a `.env` file in the project root directory and add your Groq API key:
```
GROQ_API_KEY=your_api_key_here
```

### 4. Run the Streamlit App
To launch the application, run the following command:
```bash
streamlit run app.py
```

The application will start, and you can access it in your browser.

## Functions

### `load_data(data_path)`
Loads PDF documents from a specified directory.

#### Arguments:
- `data_path`: Path to the directory containing the PDF files.

#### Returns:
- A list of loaded documents.

### `load_data_from_uploaded_pdf(file)`
Loads a single uploaded PDF file.

#### Arguments:
- `file`: File object of the uploaded PDF.

#### Returns:
- A list of loaded documents.

### `load_data_from_url(url)`
Loads data from the provided URL.

#### Arguments:
- `url`: URL to scrape data from.

#### Returns:
- A list of loaded documents.

### `text_split(data)`
Splits documents into chunks for easier processing.

#### Arguments:
- `data`: A list of documents to split.

#### Returns:
- A list of document chunks.

### `download_huggingface_embedding()`
Downloads and returns the HuggingFace embeddings model.

#### Returns:
- HuggingFaceEmbeddings object.



## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

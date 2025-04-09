from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv(override=True)
import os
import streamlit as st

# gemini api key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# configure the genai 
genai.configure(api_key=GEMINI_API_KEY)


def get_pdf_text(pdf_docs):
    """
    Extracts text content from a list of PDF files.

    Args:
        pdf_docs (list): List of uploaded PDF files.

    Returns:
        str: Combined extracted text from all pages of the PDF documents.
    """
    text = ""
    # Handle single file upload
    if not isinstance(pdf_docs, list):
        pdf_docs = [pdf_docs]
        
    for pdf in pdf_docs:
        if pdf is not None:
            # Streamlit file uploader returns a file-like object
            doc_pdf = PdfReader(pdf)
            for page in doc_pdf.pages:
                text += page.extract_text()
    return text


def get_text_chunks(text):
    """
    Splits large text into smaller, overlapping chunks for processing.

    Args:
        text (str): The full text to be split.

    Returns:
        list: A list of text chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=200)
    chunks = text_splitter.split_text(text=text)
    return chunks


embedding_model = None
def get_vector_store(text_chunks):
    """
    Creates and saves a FAISS vector store from the provided text chunks using Google's embedding model.

    Args:
        text_chunks (list): List of text chunks to embed and store.

    Returns:
        None
    """
    global embedding_model
    # Use the API key for authentication
    embedding_model = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GEMINI_API_KEY
    )
    vectore_store = FAISS.from_texts(text_chunks, embedding_model)
    vectore_store.save_local(folder_path="faiss-index")


def get_conversational_chain():
    """
    Sets up a conversational retrieval QA chain with a custom prompt and Gemini language model.

    Returns:
        Chain: A LangChain question-answering chain ready to process queries using the given context.
    """
    prompt_template = """You are a knowledgeable and professional AI assistant specializing in providing accurate information from the given context. "
                         "Your role is to:\n\n"
                         "1. Provide clear, concise, and accurate answers based solely on the provided context\n"
                         "2. If the context doesn't contain enough information to fully answer a question, acknowledge this limitation\n"
                         "3. Maintain a professional and helpful tone while ensuring factual accuracy\n"
                         "4. Use direct quotes from the context when relevant to support your answers\n"
                         "5. Organize complex responses in a structured, easy-to-read format\n"
                         "6. If you need to make assumptions, explicitly state them\n\n"
                         "Remember:\n"
                         "- Stay within the scope of the provided context\n"
                         "- Avoid making up information or speculating beyond the given content\n"
                         "- If multiple interpretations are possible, present them clearly\n"
                         "- Maintain consistency in your responses\n\n"
                         "Format your responses in a clear, professional manner using appropriate markdown formatting when helpful.\n\n"
                         "Context:\n{context}\n\n"
                         "Question: {input}\n\n"
                         "Answer: 
                         """
    # Use the correct model name format for Gemini
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=GEMINI_API_KEY)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "input"])
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    return chain


def user_query(question):
    """
    Processes a user question by retrieving relevant documents from the FAISS vector store and generating a response.

    Args:
        question (str): The user's input question.

    Returns:
        dict: The generated response from the language model based on relevant context.
    """
    global embedding_model
    if embedding_model is None:
        # Recreate the embedding model if it doesn't exist
        embedding_model = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GEMINI_API_KEY
        )
    
    # Load the vector store with allow_dangerous_deserialization=True since we trust our own files
    new_db = FAISS.load_local(
        folder_path="faiss-index", 
        embeddings=embedding_model,
        allow_dangerous_deserialization=True
    )
    docs = new_db.similarity_search(query=question, k=10)

    chain = get_conversational_chain()

    # Format the input correctly with both 'input_documents' and 'input' keys
    chain_input = {
        "input_documents": docs,
        "input": question
    }
    
    response = chain(
        chain_input,
        return_only_outputs=True
    )
    st.write(response["output_text"])

       
# Chat with Multiple PDF Documents

A powerful application that allows you to chat with multiple PDF documents using LangChain and Google Gemini Pro. This tool extracts text from PDFs, processes it, and enables you to ask questions about the content, receiving AI-generated responses based on the document content.

![Chat with PDFs](https://img.shields.io/badge/Chat-PDFs-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.1.0-green)
![Google Gemini](https://img.shields.io/badge/Google-Gemini-orange)

## Features

- 📄 Upload and process multiple PDF documents simultaneously
- 🔍 Extract text content from PDFs efficiently
- 🧠 Process text using advanced NLP techniques
- 💬 Interactive chat interface with document-aware responses
- 🔄 Maintain conversation history for context-aware interactions
- 🎨 Modern, user-friendly interface built with Streamlit
- 🔒 Secure handling of API keys and sensitive information

## How It Works

1. **Document Processing**: The application extracts text from uploaded PDF documents
2. **Text Chunking**: Large documents are split into manageable chunks for processing
3. **Vector Embedding**: Text chunks are converted to vector embeddings using Google's embedding model
4. **Vector Storage**: Embeddings are stored in a FAISS vector database for efficient retrieval
5. **Question Answering**: When you ask a question, the system:
   - Retrieves relevant document chunks based on semantic similarity
   - Processes your question along with the retrieved context
   - Generates a response using Google Gemini Pro

## Installation

### Prerequisites

- Python 3.8 or higher
- Google Gemini API key

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/MohammedHamza0/chat-with-multiple-pdf-documents.git
   cd chat-with-multiple-pdf-documents
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   - Copy `.env.example` to `.env`
   - Add your Google Gemini API key to the `.env` file:
     ```
     GEMINI_API_KEY=your_api_key_here
     ```

## Usage

1. Start the application:
   ```bash
   streamlit run app.py
   ```

2. Open your web browser and navigate to the URL displayed in the terminal (typically http://localhost:8501)

3. Upload one or more PDF documents using the file uploader in the sidebar

4. Click "Process" to extract and index the content of the PDFs

5. Start asking questions about the content of your documents in the chat interface

6. Use the "Clear Chat" button to reset the conversation history

## Project Structure

```
chat-with-multiple-pdf-documents/
├── app.py                  # Main Streamlit application
├── src/
│   ├── helper.py           # Core functionality for PDF processing and QA
│   └── __init__.py         # Package initialization
├── faiss-index/            # Vector database storage
├── requirements.txt        # Project dependencies
├── .env                    # Environment variables (not in repo)
├── .env.example            # Example environment variables
└── README.md               # Project documentation
```

## Dependencies

- **streamlit**: Web application framework
- **google-generativeai**: Google's Generative AI API
- **python-dotenv**: Environment variable management
- **langchain**: Framework for LLM applications
- **PyPDF2**: PDF processing library
- **faiss-cpu**: Vector similarity search
- **langchain-google-genai**: LangChain integration with Google Gemini
- **langchain-community**: Community components for LangChain

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) for the powerful framework
- [Google Gemini](https://ai.google.dev/) for the advanced language model
- [Streamlit](https://streamlit.io/) for the web application framework
- [FAISS](https://github.com/facebookresearch/faiss) for vector similarity search

## Developer

**Mohammed Hamza Khalifa**  
AI Engineer | Passionate about machine learning, NLP, and real-world AI solutions.

[🔗 Connect on LinkedIn](https://www.linkedin.com/in/mohammed-hamza-4184b2251/)

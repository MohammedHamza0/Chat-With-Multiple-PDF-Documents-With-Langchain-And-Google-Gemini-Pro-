import streamlit as st
import os
from src.helper import (get_pdf_text, 
                        get_text_chunks, 
                        get_vector_store, 
                        get_conversational_chain, 
                        user_query)

# streamlit function
def main():
     st.set_page_config(page_title="Chat PDFs", page_icon="💬")
     st.header("Chat with Multiple PDFs")
     st.caption("Ask me any question related to the uploaded files..")
     user_question = st.chat_input(placeholder="inter your question.")
     if user_question:
         user_query(user_question)
          
          
     with st.sidebar:
          st.title("Menu:")
          doc_files = st.file_uploader(label="Upload your PDF files")
          if st.button("Process"):
               with st.spinner("Processing...."):
                    raw_text = get_pdf_text(doc_files)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Done.") 
     
     
if __name__ == "__main__":
     main()

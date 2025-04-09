import streamlit as st
from src.helper import (get_pdf_text, 
                        get_text_chunks, 
                        get_vector_store, 
                        get_conversational_chain, 
                        user_query)

# streamlit function
def main():
     st.set_page_config(page_title="Chat PDFs", page_icon="💬")
     # Custom CSS styling
     st.markdown("""
          <style>
          .chat-container {
               width: 100%;
               max-width: 700px;
               margin: 0 auto;
               background-color: #131722;
               padding: 20px;
               border-radius: 8px;
               box-shadow: 0 4px 8px rgba(0,0,0,0.1);
          }
          .chat-message {
               margin-bottom: 12px;
               padding: 10px;
               border-radius: 8px;
               box-shadow: 0 2px 4px rgba(0,0,0,0.1);
          }
          .chat-message.user {
               background-color: #0a5e2a;
               text-align: right;
               flex-direction: row-reverse;
               color: #e8ebf1;
          }
          .chat-message.bot {
               background-color: #e1f7d5;
               text-align: left;
               flex-direction: row;
               color: #131722;
          }
          .chat-avatar {
               width: 40px;
               height: 40px;
               border-radius: 50%;
               margin-right: 10px;
          }
          .header {
               text-align: center;
               padding: 10px;
               background-color: #1E90FF;
               color: white;
               border-radius: 8px;
          }
          .stTextInput>div>div>input {
               font-size: 16px;
               background-color: #1d2330;
               color: #e8ebf1;
          }
          body {
               color: #e8ebf1;
               background-color: #131722;
          }
          </style>
     """, unsafe_allow_html=True)

     # Avatars
     bot_avatar = "https://img.freepik.com/free-vector/graident-ai-robot-vectorart_78370-4114.jpg"
     user_avatar = "https://cdn-icons-png.freepik.com/512/6596/6596121.png"
     
     # Initialize history
     if "history" not in st.session_state:
        st.session_state.history = []
        
     st.header("Chat with Multiple PDFs")
     st.caption("Ask me any question related to the uploaded files..")
     user_question = st.chat_input(placeholder="inter your question.")
     if user_question:
          response = user_query(user_question)
          st.session_state.history.append({"role": "user", "message": user_question})
          st.session_state.history.append({"role": "bot", "message": response})
          
     # Clear history
     if st.button("🧹 Clear Chat"):
          st.session_state.history = []
          st.rerun()
          
     # Display chat history
     with st.container():
          for chat in st.session_state.history:
               if chat["role"] == "user":
                    st.markdown(f'''
                         <div class="chat-message user">
                         <img class="chat-avatar" src="{user_avatar}" alt="User Avatar"/>
                         <div>{chat["message"]}</div>
                         </div>
                    ''', unsafe_allow_html=True)
               else:
                    st.markdown(f'''
                         <div class="chat-message bot">
                         <img class="chat-avatar" src="{bot_avatar}" alt="Doctor Avatar"/>
                         <div>{chat["message"]}</div>
                         </div>
                    ''', unsafe_allow_html=True)
               
     with st.sidebar:
          st.title("Menu:")
          doc_files = st.file_uploader(label="Upload your PDF files", accept_multiple_files=True)
          if st.button("Process"):
               with st.spinner("Processing...."):
                    raw_text = get_pdf_text(doc_files)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Done.") 
                    
                    
     with st.sidebar:
          st.markdown("""
                         **Developer:**
                         **Mohammed Hamza Khalifa**  
                         AI Engineer | Passionate about machine learning, NLP, and real-world AI solutions.
                         
                         [🔗 Connect on LinkedIn](https://www.linkedin.com/in/mohammed-hamza-4184b2251/)
                         ---
                         © 2025 Mohammed Khalifa
                    
                    """)
     # Auto scroll
     st.markdown('<script>window.scrollTo(0, document.body.scrollHeight);</script>', unsafe_allow_html=True)

     
     
if __name__ == "__main__":
     main()

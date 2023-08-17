import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import TextLoader
from langchain.document_loaders import Docx2txtLoader
import os
import tempfile


DB_FAISS_PATH = 'vectorstore/db_faiss'

st.sidebar.title("Document Processing")

uploaded_files = st.sidebar.file_uploader("Upload files", accept_multiple_files=True)

text = []
if uploaded_files:
    for file in uploaded_files:
        file_extension = os.path.splitext(file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(file.read())  # Save the uploaded file to a temporary location
            temp_file_path = temp_file.name
        if file_extension == ".pdf":
            loader = PyPDFLoader(temp_file_path)  # Pass the temporary file path
            text.extend(loader.load())
        elif file_extension == ".docx" or file_extension == ".doc":
            loader = Docx2txtLoader(temp_file_path)  # Pass the temporary file path
            text.extend(loader.load())
        elif file_extension == ".txt":
            loader = TextLoader(temp_file_path)  # Pass the temporary file path
            text.extend(loader.load())
        os.remove(temp_file_path)  # Remove the temporary file
    #split text into chunks
    text_splitter = CharacterTextSplitter(separator= "\n",chunk_size=500, chunk_overlap=100,length_function=len)
    text_chunks = text_splitter.split_documents(text)


    #create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                   model_kwargs={'device':"cpu"})

    #Create vector store
    vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)
    vector_store.save_local(DB_FAISS_PATH)

    #create llm
    llm = CTransformers(model="llama-2-7b-chat.ggmlv3.q4_0.bin",model_type="llama",
                    config={'max_new_tokens':500,'temperature':0.01})

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    chain = ConversationalRetrievalChain.from_llm(llm=llm,chain_type='stuff',
                                              retriever=vector_store.as_retriever(search_kwargs={"k":2}),
                                              memory=memory)

    st.title("Multi-Docs ChatBot using llama2 :books:")

    def conversation_chat(query):
        result = chain({"question": query, "chat_history": st.session_state['history']})
        st.session_state['history'].append((query, result["answer"]))
        return result["answer"]

    def initialize_session_state():
        if 'history' not in st.session_state:
            st.session_state['history'] = []

        if 'generated' not in st.session_state:
            st.session_state['generated'] = ["Hello! Ask me anything about 🤗"]

        if 'past' not in st.session_state:
            st.session_state['past'] = ["Hey! 👋"]

    def display_chat_history():
        reply_container = st.container()
        container = st.container()

        with container:
            with st.form(key='my_form', clear_on_submit=True):
                user_input = st.text_input("Question:", placeholder="Ask about your Documents", key='input')
                submit_button = st.form_submit_button(label='Send')

            if submit_button and user_input:
                output = conversation_chat(user_input)

                st.session_state['past'].append(user_input)
                st.session_state['generated'].append(output)

        if st.session_state['generated']:
            with reply_container:
                for i in range(len(st.session_state['generated'])):
                    message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                    message(st.session_state["generated"][i], key=str(i), avatar_style="fun-emoji")

    #Initialize session state
    initialize_session_state()
    #Display chat history
    display_chat_history()
import os
import pickle
import streamlit as st
from dotenv import load_dotenv
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import AzureOpenAI, HuggingFaceHub
from langchain.chat_models import AzureChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplate import bot_template, user_template, css

def get_pdf_text(pdf_docs):
    text = ""
    for pdf_doc in pdf_docs:
        pdf_reader = PdfReader(pdf_doc)
        for page in pdf_reader.pages:
            text += page.extract_text()
        store_name = pdf_doc.name[:-4]
        st.session_state.store.append(store_name)

    return text



def get_text_chunks_recursively(raw_text):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n", "\r\n", "\r", "\t", " "],
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text=raw_text)
    return chunks


def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text=raw_text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings(
        deployment=os.getenv("AZURE_OPENAI_API_EMBEDDINGS_DEPLOYMENT_NAME") or '',
        model=os.getenv("AZURE_OPENAI_API_EMBEDDINGS_MODEL_NAME") or '',
        # openai_api_base=os.getenv("AZURE_OPENAI_API_BASE") or '',
        # openai_api_type=os.getenv("AZURE_OPENAI_API_TYPE") or 'azure',
        chunk_size=1,
        client=None,
    )        
    VectorStore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)

    # with open(f"{st.store_name}.pkl", "wb") as f:
    #     pickle.dump(VectorStore, f)    

    return VectorStore
    

def get_vectorstore_fromHugging(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    VectorStore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)

    return VectorStore


def get_conversation_chain(vector_store):
    # llm = AzureOpenAI(
    #     deployment_name=os.getenv("AZURE_OPENAI_API_COMPLETIONS_DEPLOYMENT_NAME") or '',
    #     client=None,
    #     model=os.getenv("AZURE_OPENAI_API_COMPLETIONS_MODEL_NAME") or '',
    #     temperature=0.9,
    # )
    llm_chat = AzureChatOpenAI(
        deployment_name=os.getenv("AZURE_OPENAI_API_CHAT_DEPLOYMENT_NAME") or '',
        client=None,
        temperature=0.5,
    )
    # llm_hugging = HuggingFaceHub(client=None, repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
    
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm_chat, 
        # llm=llm_hugging,
        retriever=vector_store.as_retriever(),
        memory=memory
    )
    return conversation_chain
     

def handle_user_input(user_question):
    response = st.session_state.conversation({"question": user_question})
    st.session_state.history = response["chat_history"]

    for i, message in enumerate(st.session_state.history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    load_dotenv()    
    log_level = os.getenv("LOG_LEVEL", "INFO")
    st.set_page_config(page_title="Docs Analytics Chat Bot", page_icon=":female-doctor:")

    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "store" not in st.session_state:
        st.session_state.store = []

    st.header("Chat with Medical Practice Assistant :female-doctor:")
    user_question = st.text_input("Ask questions about the patient :male-pilot:")
    
    # if user_question:

    #     if st.session_state.vector_store:
    #         docs = st.session_state.vector_store.similarity_search(query=user_question, k=3)
    #         if (log_level == "TRACE"):
    #             # st.write(docs)
    #             st.write("Similarity search completed")
    
    #         # LLM
    #         llm = AzureOpenAI(
    #             deployment_name=os.getenv("AZURE_OPENAI_API_COMPLETIONS_DEPLOYMENT_NAME") or '',
    #             client=None,
    #             model=os.getenv("AZURE_OPENAI_API_COMPLETIONS_MODEL_NAME") or '',
    #             temperature=0.9,
    #         )
    #         chain = load_qa_chain(llm=llm, chain_type="stuff")
    #         with get_openai_callback() as cb:
    #             response = chain.run(input_documents=docs, question=user_question)
    #             print(cb)
    #         st.write(response)
    
    
    if user_question and st.session_state.conversation:
        handle_user_input(user_question)

    #Sidebar content
    with st.sidebar:
        st.subheader("Documents :books:")
        pdf_docs = st.file_uploader(
            "Upload PDF documents here and click on 'Process'", type=["pdf"], accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing..."):
                # get the text from the PDF
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)
                
                # create vector store
                st.session_state.vector_store = get_vectorstore(text_chunks)
                
                # create conversation chain
                st.session_state.conversation = get_conversation_chain(st.session_state.vector_store)
                
            if (log_level == "TRACE"):
                st.write("Embeddings computation completed")

        add_vertical_space(5)        
        st.markdown('''
        ## About
        This app is an LLM-powered chatbot built to analyze documents and answer questions about them.
        It is powered by:
        - [Streamlit](https://streamlit.io)
        - [LangChain](https://python.langchain.com/)
        - [Azure OpenAI](https://azure.microsoft.com/en-us/products/cognitive-services/openai-service/)
        
        ''')
        add_vertical_space(5)
        st.write("Made by Danny Garber, 2023")

     
    
    
            
if __name__ == "__main__":
    main()
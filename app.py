import os
import pickle
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_chat import message
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import AzureOpenAI
from langchain.chat_models import AzureChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplate import bot_template, user_template, css



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


def get_vectorstore(text_chunks, *args, **kwargs):
    embeddings = OpenAIEmbeddings(
        deployment=os.getenv("AZURE_OPENAI_API_EMBEDDINGS_DEPLOYMENT_NAME") or '',
        model=os.getenv("AZURE_OPENAI_API_EMBEDDINGS_MODEL_NAME") or '',
        # openai_api_base=os.getenv("AZURE_OPENAI_API_BASE") or '',
        # openai_api_type=os.getenv("AZURE_OPENAI_API_TYPE") or 'azure',
        chunk_size=1,
        client=None,
    )        
    VectorStore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)

    file_name = kwargs.get("file", None)
    if file_name:
        cur_path = os.path.dirname(__file__)
        pkl_file = os.path.join(cur_path, "docs", f"{file_name}.pkl")
        with open(pkl_file, "wb") as f:
            pickle.dump(VectorStore, f)    
            st.write("Created new embeddings vector and saved to the local disk")        

    return VectorStore
    



def get_conversation_chain(vector_store):
    llm = AzureOpenAI(
        deployment_name=os.getenv("AZURE_OPENAI_API_COMPLETIONS_DEPLOYMENT_NAME") or '',
        client=None,
        model=os.getenv("AZURE_OPENAI_API_COMPLETIONS_MODEL_NAME") or '',
        temperature=0.7,
    )
    llm_chat = AzureChatOpenAI(
        deployment_name=os.getenv("AZURE_OPENAI_API_CHAT_DEPLOYMENT_NAME") or '',
        client=None,
        temperature=0.5,
    )
    
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm_chat, 
        retriever=vector_store.as_retriever(),
        memory=memory,
        callbacks=get_openai_cost()
    )
    return conversation_chain


     
def get_openai_cost():
    with get_openai_callback() as cb:
        print(cb)



def create_conversation_chain_from_text(text):
    # get the text chunks
    text_chunks = get_text_chunks_recursively(text)
    
    # create vector store
    if st.session_state.vector_store is None:
        st.session_state.vector_store = get_vectorstore(text_chunks)
    else:
        st.session_state.vector_store.merge_from(get_vectorstore(text_chunks))
    
    # create conversation chain
    return get_conversation_chain(st.session_state.vector_store)


def create_conversation_chain_from_pdf(pdf_docs):
    # get the text chunks
    for pdf_doc in pdf_docs:
        pdf_reader = PdfReader(pdf_doc)
        
        # check if the vector store exists
        file_name = pdf_doc.name[:-4]
        cur_path = os.path.dirname(__file__)
        pkl_file = os.path.join(cur_path, "docs", f"{file_name}.pkl")

        if os.path.exists(pkl_file):
            # with open(f"{file_name}.pkl", "rb") as f:
            with open(pkl_file, "rb") as f:
                st.write("Loaded existing embeddings vector from the local disk for ", file_name, "")
                vector = pickle.load(f)
                st.session_state.vector_store.merge_from(vector)
        else:
            text = ""
        
            for page in pdf_reader.pages:
                text += page.extract_text()

            text_chunks = get_text_chunks_recursively(text)
    
            # create vector store
            vector = get_vectorstore(text_chunks, file=file_name)
            st.session_state.vector_store.merge_from(vector)
    
    # create conversation chain
    return get_conversation_chain(st.session_state.vector_store)


def handle_user_input(user_question, show_response=True):
    response = st.session_state.conversation({"question": user_question})

    if show_response:
        st.session_state.history = response["chat_history"]
        for i, message in enumerate(st.session_state.history, start=2):
            if i % 2 == 0:
                st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)


def process_user_input(user_question):
    if st.session_state.conversation:
        response = st.session_state.conversation({"question": user_question})
        st.session_state.history = response["chat_history"]
        formatted_response = bot_template.replace("{{MSG}}", response["answer"])
        return response["answer"]
        # return st.session_state.history[-1].content
    
    st.write("No conversation history found. Please upload a PDF file or enter a text to start a conversation.")
    return None

    
    

def main():
    load_dotenv()    
    log_level = os.getenv("LOG_LEVEL", "INFO")
    st.set_page_config(page_title="Docs Analytics Chat Bot", page_icon=":female-doctor:")

    st.write(css, unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    st.header("Chat with Medical Practice Assistant :female-doctor:")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if st.session_state.vector_store is None:
        system_prompt = "You specialize in medical practice and healthcare insurance plan eligibility verification. " \
            "Your objectives are to determine if the patient's insurance plan is in network or requires prior referral. " \
            "Before you response, please read the document carefully and make sure you understand the question. " \
            "If you find that the patient's insurance is not in the list of insurances that we are NOT in network with, respond that we are in network with patient's insurance." \
            "If you find that the patient's insurance is not in the list of insurances we require referral, respond that we are in network with patient's insurance and no referral is required." \
            "If you can't determine the answer, you should respond that the documents provided do not contain enough information to determine the insurance eligibility of the patient."

        st.session_state.conversation = create_conversation_chain_from_text(system_prompt)    
        
    
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

    if user_question := st.chat_input("Please type here your question"):
        st.session_state.messages.append({"role": "user", "content": user_question})
        # if st.session_state.conversation:
        with st.chat_message("user"):
            st.markdown(user_question)

        full_response = ""
        with st.chat_message("assistant"):
            full_response = process_user_input(user_question)
            st.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
   

    #Sidebar content
    with st.sidebar:
        st.subheader("Documents :books:")
        pdf_docs = st.file_uploader(
            "Upload PDF documents here and click on 'Process'", type=["pdf"], accept_multiple_files=True)

        if st.button("Process"):
            if pdf_docs is None:
                st.warning("Please upload at least one PDF file first.")
            else:
                with st.spinner("Processing..."):
                    # create conversation chain from the pdf documents
                    st.session_state.conversation = create_conversation_chain_from_pdf(pdf_docs)

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
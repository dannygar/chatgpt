import os
import streamlit as st
from dotenv import load_dotenv
import pickle
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback


#Sidebar content
with st.sidebar:
    st.title("Docs Analytics Chat Bot")
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


load_dotenv()    
def main():
    st.header("Chat with Docs Analytics Bot")
    
    log_level = os.getenv("LOG_LEVEL", "INFO")
    
    # upload a PDF file
    pdf = st.file_uploader("Upload a PDF file", type=["pdf"])

    if pdf is not None:
        # read the PDF file
        pdf_reader = PdfReader(pdf)
        
        #st.write(chunks)
        
        # embeddings
        store_name = pdf.name[:-4]

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
            if (log_level == "TRACE"):
                st.write("Loaded existing embeddings vector from the local disk")
        else:
            if (log_level == "TRACE"):
                st.write("Created new embeddings vector and saved to the local disk")
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
        
            # st.write(text)
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_text(text=text)
            embeddings = OpenAIEmbeddings(
                deployment=os.getenv("AZURE_OPENAI_API_EMBEDDINGS_DEPLOYMENT_NAME"),
                model=os.getenv("AZURE_OPENAI_API_EMBEDDINGS_MODEL_NAME"),
                chunk_size=1
            )        
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)    
            if (log_level == "TRACE"):
                st.write("Embeddings computation completed")

        # Accept user questions/queries
        query = st.text_input("Ask questions about the document:")
        if query:
            docs = VectorStore.similarity_search(query=query, k=3)
            if (log_level == "TRACE"):
                st.write(docs)
                st.write("Similarity search completed")
        
            # LLM
            llm = OpenAI(
                deployment_id=os.getenv("AZURE_OPENAI_API_COMPLETIONS_DEPLOYMENT_NAME"),
                model_name=os.getenv("AZURE_OPENAI_API_COMPLETIONS_MODEL_NAME"),
                # deployment_id="gpt35-turbo",
                # model_name="gpt-3.5-turbo",
                temperature=0.9,
            )
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                # response = chain.run(input_documents=docs, question=query)
                response = chain.run(input_documents=docs, question=query)
                print(cb)
            st.write(response)
            
if __name__ == "__main__":
    main()
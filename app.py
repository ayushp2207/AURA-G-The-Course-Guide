import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

api_key = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=api_key)

def get_pdf_text(pdf_file):
    text = ""
    pdf_reader = PdfReader(pdf_file)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def create_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context and use your own tools also and describe the context and provide related answer. Make sure to provide all the details. If the person asks you about the name of the faculty, GIVE A VERY DETAILED INTERACTIVE ANSWER, like respond with "The course will be taught by Professor xyz". If the person asks you about the schedule, make sure to respond in a table format with weekdays clearly mentioned. In case you don't find the course, make sure to respond with an apology. 
    If the answer is not in the provided context, just say, "The answer is not available in the context." Do not provide a wrong answer.\n\n
    Context:\n {context}\n
    Question: \n{question}\n
    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def process_question(vector_store, user_question):
    chain = get_conversational_chain()
    docs = vector_store.similarity_search(user_question)
    
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    return response["output_text"]

def main():
    st.set_page_config(page_title="Course Information Chatbot", page_icon="🎓")
    st.header("Course Information Chatbot 📚")

    pdf_file = st.file_uploader("Upload your course PDF here", type="pdf")

    if pdf_file is not None:
        text = get_pdf_text(pdf_file)
        text_chunks = get_text_chunks(text)
        vector_store = create_vector_store(text_chunks)

        user_question = st.text_input("Ask a question about the course:")

        if user_question:
            response = process_question(vector_store, user_question)
            st.write("Reply:")
            st.write(response)

if __name__ == "__main__":
    main()

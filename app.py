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

def get_pdf_text_from_file(file_path):
    text = ""
    with open(file_path, 'rb') as pdf_file:
        pdf_reader = PdfReader(pdf_file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def create_vector_store_for_pdf(pdf_path):
    text = get_pdf_text_from_file(pdf_path)
    chunks = get_text_chunks(text)
    
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

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

def user_input(user_question, course_name):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Load the vector store
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    # Find the relevant document based on course name
    docs = new_db.similarity_search(course_name)

    chain = get_conversational_chain()
    
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    return response["output_text"]

# Streamlit application
def main():
    st.title("Course Information Assistant")

    directory = "./pdfs"  # Replace with the actual path to your PDF directory
    course_name = st.text_input("Enter the course name:")

    if course_name:
        course_name = course_name.replace(" ", "").lower()
        
        matching_pdf = None
        for filename in os.listdir(directory):
            if filename.endswith(".pdf"):
                normalized_filename = filename.replace(" ", "").lower()
                if course_name in normalized_filename:
                    matching_pdf = os.path.join(directory, filename)
                    break
        
        if matching_pdf:
            create_vector_store_for_pdf(matching_pdf)
            user_question = st.text_input("Ask a question about the course:")
            if user_question:
                response = user_input(user_question, course_name)
                st.write("Reply: ", response)
        else:
            st.write("No matching course found.")

if __name__ == "__main__":
    main()

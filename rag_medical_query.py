import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
import torchvision

# Load environ variable
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Load the WHO_SelfCare.pdf document
loader = PyPDFLoader(r"E:\RAG\self_care_doc.pdf")
documents = loader.load()

# Split the document in chunk
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = splitter.split_documents(documents)

DB_NAME = "vector_db"
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)
# Add documents to vectorstore
vectorstore.add_documents(docs)

# Persist the database
vectorstore.persist()

# Retrieve the document
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Prompt builder
def generate_prompt(context, user_query):
    return f"""
    You are acting as a responsible virtual medical assistant.
    Your task is to answer user health queries using ONLY the information
    provided in the retrieved context. Do not invent or assume details
    outside the context.

    Context:
    {context}

    User Query:
    {user_query}

    Guidelines for your response:
    1. Read the context carefully and extract the most relevant information
       that addresses the user’s query.
    2. Present the answer in clear, concise bullet points.
    3. Each bullet point should explain one actionable step, recommendation,
       or fact in simple language.
    4. If the context provides multiple options, list them all in separate points.
    5. If the context does not fully answer the query, state that clearly
       instead of guessing.
    6. Always conclude with this disclaimer:
       "This information is for educational/demo purposes only.
        Please consult a licensed medical professional for actual medical advice."

    """

# LLM call
def generate_medical_advice(prompt):
    GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
    google_api_key = os.getenv("GOOGLE_API_KEY")
    llm = ChatOpenAI(model='gemini-3.1-flash-lite',
                     api_key=google_api_key,
                     base_url=GEMINI_BASE_URL)
    response = llm.invoke(prompt)
    return response.content.strip()



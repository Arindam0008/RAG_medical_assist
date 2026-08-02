
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# Load environ variable
load_dotenv()

# Load the WHO_SelfCare.pdf document
loader = PyPDFLoader(r"E:\RAG\self_care_doc.pdf")
documents = loader.load()

# Split the document in chunk
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = splitter.split_documents(documents)

# Build FAISS index
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.from_documents(docs, embeddings)

# # Save the embeddings
# vectorstore.save_local("faiss_index")
# # Load retriever
# vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# Retrieve the document
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

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
    llm = ChatOpenAI(model="gpt-4o-mini")
    response = llm.invoke(prompt)
    return response.content.strip()



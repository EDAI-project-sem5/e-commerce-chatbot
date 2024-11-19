# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
import os

# Initialize environment variables for LangChain setup
os.environ["LANGCHAIN_TRACING_V2"] = ""
os.environ["LANGCHAIN_API_KEY"] = ""
os.environ["LANGCHAIN_ENDPOINT"] = ""
os.environ["LANGCHAIN_PROJECT"] = ""
os.environ["GOOGLE_API_KEY"] = ""     

# Define embeddings and model
gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", convert_system_message_to_human=True)

# Load and split documents
loader = WebBaseLoader(web_paths=("https://swadeshshop13.blogspot.com/2024/11/swadeshshop-ultimate-shopping.html",))
doc = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(doc)

# Create vector store and retrieval chain
vectorstore = Chroma.from_documents(documents=splits, embedding=gemini_embeddings)
retriever = vectorstore.as_retriever()

# Define system prompt and chat prompt
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "Answer such that you are the company representative. "
    "If you don't know the answer, say that you don't know. "
    "Use three sentences maximum and keep the answer concise."
    "Don't skip any important information from the answer."
    "\n\n"
    "{context}"
)

from langchain_core.prompts import ChatPromptTemplate
chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Set up question-answering chain
question_answering_chain = create_stuff_documents_chain(model, chat_prompt)
rag_chain = create_retrieval_chain(retriever, question_answering_chain)

# FastAPI initialization
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI
app = FastAPI()

# Allow CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # Frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Define the input data model
class ChatRequest(BaseModel):
    input: str

# Define the response model
class ChatResponse(BaseModel):
    answer: str

# Chat endpoint
@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # Invoke the RAG model with input from frontend
        response = rag_chain.invoke({"input": request.input})
        answer = response.get("answer", "Sorry, I couldn't retrieve an answer.")
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Root endpoint to check API status
@app.get("/")
def read_root():
    return {"message": "Welcome to the Chat API"}

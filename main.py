import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

# --- STEP 0: Load environment variables ---
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# ✅ Initialize the Groq LLM (replace with your real model name, e.g., "llama-3.1-70b-versatile")
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

# --- STEP 1: Load and Split PDF ---
pdf_path = input("Enter path to your PPT or PDF file: ").strip()
loader = PyPDFLoader(pdf_path)
pages = loader.load()

# Split into manageable text chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
docs = text_splitter.split_documents(pages)

# --- STEP 2: Create and persist Chroma DB ---
persist_directory = "./chroma_store"
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

vectordb = Chroma.from_documents(
    documents=docs,
    embedding=embedding_function,
    persist_directory=persist_directory
)

print(f"✅ Database created and saved in: {persist_directory}")

# --- STEP 3: Simple Q&A Loop ---
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

while True:
    query = input("\nAsk me anything about your document (or type 'exit'): ").strip()
    if query.lower() in ["exit", "quit"]:
        print("Goodbye! 👋")
        break

    # Retrieve top 3 relevant chunks
    context_docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in context_docs])

    # Build LLM prompt
    prompt = f"""
You are a helpful teacher-scholar assistant.
Answer the following question using the context below.

Context:
{context}

Question: {query}
Answer:
"""

    # Generate and print answer
    response = llm.invoke(prompt)
    print("\n🧠 Answer:\n", response.content)

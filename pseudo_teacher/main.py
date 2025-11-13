import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma


load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)


pdf_path = input("Enter path to your PPT or PDF file: ").strip()
loader = PyPDFLoader(pdf_path)
pages = loader.load()


text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
docs = text_splitter.split_documents(pages)


persist_directory = "./chroma_store"
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

vectordb = Chroma.from_documents(
    documents=docs,
    embedding=embedding_function,
    persist_directory=persist_directory
)

print(f"Database created and saved in: {persist_directory}")

retriever = vectordb.as_retriever(search_kwargs={"k": 3})

while True:
    query = input("Ask me anything about your document (or type 'exit'): ").strip()
    if query.lower() in ["exit", "quit"]:
        print("bye!")
        break

    context_docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in context_docs])

    prompt = f"""
You are a helpful teacher-scholar assistant.
Answer the following question using the context below.

Context:
{context}

Question: {query}
Answer:
"""

    response = llm.invoke(prompt)
    print("Answer:\n", response.content)
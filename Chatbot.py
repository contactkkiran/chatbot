import os
from dotenv import load_dotenv
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# Load environment variables from .env file
load_dotenv()

# Sample documents 
# We’ll load some text documents, split them into chunks, and embed them into ChromaDB.
docs = ["Playwright is a modern automation framework.", 
        "Selenium is older but widely used."]

# Split into chunks
splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
documents = splitter.create_documents(docs)

# Create embeddings
# OpenAI API key is automatically loaded from OPENAI_API_KEY environment variable
embeddings = OpenAIEmbeddings()

# Store in ChromaDB
vectorstore = Chroma.from_documents(documents, embeddings, persist_directory="./chroma_db")
vectorstore.persist()
query = "What is Playwright?"
results = vectorstore.similarity_search(query, k=2)
for r in results:
    print(r.page_content)

# ---- Reload persisted DB ----
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# ---- Convert to retriever ----
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# ---- Retrieve relevant docs ----
docs = retriever.invoke(query)
print("\n🔍 Retrieved Docs:")
for d in docs:
    print(d.page_content)

# ---- Pass to LLM (RAG step) ----
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")

context = "\n".join([d.page_content for d in docs])

prompt = f"""
Answer the question based only on the context below.

Context:
{context}

Question:
{query}
"""

response = llm.invoke(prompt)

print("\n🤖 Final Answer:")
print(response.content)

#Please explain the code above in detail.
#The code above demonstrates how to use the ChromaDB vector store with LangChain to store and manage document embeddings. 
# Here's a detailed explanation of each part of the code:

# 1. **Importing Libraries**: The code imports necessary classes from the `langchain_text_splitters`, `langchain_openai`, and `langchain_community.vectorstores` modules. These classes are used for splitting text, creating embeddings, and managing the vector store, respectively.
# 2. **Sample Documents**: A list of sample documents is created, which contains two strings about Playwright and Selenium. These documents will be processed and stored in the vector store.
# 3. **Text Splitting**: An instance of `CharacterTextSplitter` is created with a specified `chunk_size` of 100 characters and `chunk_overlap` of 0. The `create_documents` method is called on the splitter instance, which takes the list of documents
# Next, the documents are split into smaller chunks based on the specified chunk size. This is useful for managing large documents and ensuring that the embeddings are created for manageable pieces of text.
# 4. **Creating Embeddings**: An instance of `OpenAIEmbeddings` is created, which will be used to generate embeddings for the document chunks. This class interacts with OpenAI's API to create vector representations of the text.
# 5. **Storing in ChromaDB**: The `Chroma.from_documents` method is called to create a vector store from the document chunks and their corresponding embeddings. The `
# persist_directory` parameter specifies the directory where the ChromaDB will be stored. The `persist` method is then called to save the vector store to disk, allowing it to be loaded and used later without needing to recreate the embeddings.
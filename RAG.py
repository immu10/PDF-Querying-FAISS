import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
import langchain
from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer
import asyncio
from langchain_text_splitters import CharacterTextSplitter
from uuid import uuid4
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import DirectoryLoader
import os

def loadPDF(dir="docs"):
    loader = DirectoryLoader(dir,glob="**/*.pdf",show_progress=True,use_multithreading=True)
    docs = loader.load()
    return docs

from langchain_text_splitters import RecursiveCharacterTextSplitter

def splitter(document):
    docs = []

    # print("\n\n\n\n\n")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    texts = text_splitter.split_documents(document)
    # print(texts)

    return texts

    
def indexMaker(embeddings):
    index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))

    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

    pages =loadPDF("docs")
    documents = splitter(pages)
  
    uuids = [str(uuid4()) for _ in range(len(documents))]

    vector_store.add_documents(documents=documents, ids=uuids)

    vector_store.save_local("faiss_index")


def indexLoader(embeddings,quest):
    vector_store = FAISS.load_local(
        "faiss_index", embeddings, allow_dangerous_deserialization=True
    )

    # docs = vector_store.similarity_search("qux")


    results = vector_store.similarity_search(
        quest,
        k=2,
        # filter={"source": "tweet"},
    )
    for res in results:
        print("results: ")
        print(f"* {res.page_content} [{res.metadata}]")
        print("\n\n\n")

    return results

def gpt4(vecResult,quest):

    openai_api_key = os.getenv("OPENAI_API_KEY") 


    prompt = f"""

            youre supposed to use the retrieved chunks to form a response relevant to the question, please only try to answer the question with preferbly 
            nothing beyond the question, oh and try to accurately reply to the question
            question: {quest}
            chunks: { vecResult}

    """
    llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=openai_api_key, 
    )
    reply = llm.invoke(prompt)
    return reply





embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
# question = input("")
question = "what is the go box"





# indexMaker(embeddings)
vecResult = indexLoader(embeddings,question)

response = gpt4(vecResult,question)



print(response.content)
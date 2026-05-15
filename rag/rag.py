from dotenv import load_dotenv
from utils import save_as_text

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

import time


load_dotenv()

def rag_pipeline(extracted_text_list:list):

    print("RAG started...")
    time.sleep(2)

    # ----------------------- initializing LLM -------------------

    llm = llm = ChatGroq(
        model='llama-3.3-70b-versatile',
        temperature=0.7
    )
    # ---------------------- text splitting --------------------

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50
    )

    documents = []

    for page_no, text in enumerate(extracted_text_list, start=1):
        chunks = splitter.split_text(text)

        for chunk in chunks:
            doc = Document(
                page_content=chunk,
                metadata={
                    "page": page_no,
                    "source": chunk
                }
            )

            documents.append(doc)

    # ----------------- embedding and vector DB ------------

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"  # Small, fast, free
    )

    vectorstore = FAISS.from_documents(
        documents,
        embeddings
    )


    vectorstore.save_local("faiss_index")
    vectorstore = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # --------------- input question and retrieved docs --------

    query = input("Ask what you want to know from the scanned document: ")
    retrieved_docs = retriever.invoke(query)

    source = retrieved_docs[0].metadata

    # -------------- prompt template -------------------

    context = "\n\n".join(
        [doc.page_content for doc in retrieved_docs]
    )

    prompt = f"""
    Answer the question using the context below.
    Don't try to generate new answer out of the context, means don't hallucinate at all.

    Context:
    {context}

    Question:
    {query}
    """

    response = llm.invoke(prompt)

    print(f"QUESTION : {query}")
    print(f"ANSWER: {response.content}")
    print(f"SOURCE: {source}")

    # ----------------- save file ------------------

    saved_file = f"""QUESTION: {query} \n
ANSWER: {response.content} \n
SOURCE: {source}
"""
    
    save_as_text(saved_file, "./output/rag_output.txt")

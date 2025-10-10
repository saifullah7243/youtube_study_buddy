# Import libraries
import os
from langsmith import traceable
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableParallel
from langchain.schema.output_parser import StrOutputParser
from dotenv import load_dotenv
from BackEnd.src.helper import extract_youtube_id, fetch_transcript, translate_to_english
load_dotenv()


# =========================  LLM & Embeddings  =========================
def init_llm(model_name="gpt-4.1-nano"):
    return init_chat_model(model_name, model_provider="openai")


def init_embeddings(model="text-embedding-3-small"):
    return OpenAIEmbeddings(model=model)


# =========================  File Helpers  =========================
def save_transcript_to_file(text:str, file_name="youtube_transcript.txt"):
    with open(file_name, "w", encoding="utf-8") as f:
        f.write(text)
    return file_name

## text loader
@traceable(name="load transcript file")
def load_documents(filename="youtube_transcript.txt"):
    loader = TextLoader(filename, encoding="utf-8")
    return loader.load()

# =========================  Text Processing  =========================
## splitiing text
@traceable(name="split text")
def split_text(documents, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)

# =========================  Vector Store  =========================
## Embedding generation and storing
@traceable(name="vector store")
def build_vectorstore(chunks,embeddings):
    return FAISS.from_documents(chunks, embeddings)

## Retriever
@traceable(name="retriever")
def get_retriever(vectorstore, k=2):
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})

# =========================  RAG Chain =========================
## Generation
@traceable(name="youtube_setup_pipeline")
def build_rag_chain(llm, retriever):
 
    template = """
    You are a helpful assistant.
    Answer ONLY from the provided transcript context.
    If the context is insufficient, just say you don't know.

    Context: {context}
    Question: {question}
    """
    prompt = PromptTemplate.from_template(template)
    str_output_parser = StrOutputParser()

    parallel_chain = RunnableParallel({
        "context": retriever,
        "question": RunnablePassthrough()
    })

    rag_chain = parallel_chain | prompt | llm | str_output_parser
    return rag_chain

# =========================  Full Pipeline =========================
@traceable(name="Youtube_rag_full_run")
def build_pipeline(youtube_url:str):
    """Build the entire RAG pipeline from YouTube URL."""
    # Step-1: Extract the youtube id
    video_id = extract_youtube_id(youtube_url)
    if not video_id:
        raise ValueError("Invalid Youtube URL")
    
    # Step-2: Fetch trascipt
    trascript = fetch_transcript(video_id)

     # Step 3: Translate if needed
    transcript_en = translate_to_english(trascript)

    # Step 4: Save and load transcript
    filename = save_transcript_to_file(transcript_en)
    documents = load_documents(filename)

    # Step 5: Split text into chunks
    chunks = split_text(documents)

    # Step 6: Create embeddings + vectorstore
    embeddings = init_embeddings()
    vectorstore = build_vectorstore(chunks, embeddings)

    # Step 7: Create retriever
    retriever = get_retriever(vectorstore)

    # Step 8: Initialize LLM & RAG chain
    llm = init_llm()
    rag_chain = build_rag_chain(llm, retriever)

    return rag_chain


rag_chain = build_pipeline("https://www.youtube.com/watch?v=fZM3oX4xEyg")

# Ask a question
config = {
    "run_name": "youtube_rag_query"
}

answer = rag_chain.invoke("What is this video about?", config=config)
print(answer)


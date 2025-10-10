# ========================== Imports ==========================
import os
from dotenv import load_dotenv

from langsmith import traceable

from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# Custom helpers
from BackEnd.src.helper import extract_youtube_id, fetch_transcript, translate_to_english

load_dotenv()

# ========================== Global Cache ==========================
RETRIEVER_CACHE = {}   # video_id -> retriever object

# ========================== LLM & Embeddings ==========================
def init_llm(model_name="gpt-4o-mini"):
    return init_chat_model(model_name, model_provider="openai")

def init_embeddings(model="text-embedding-3-small"):
    return OpenAIEmbeddings(model=model)

# ========================== File Helpers ==========================
def save_transcript_to_file(text: str, file_name="youtube_transcript.txt"):
    with open(file_name, "w", encoding="utf-8") as f:
        f.write(text)
    return file_name

@traceable(name="load_transcript_file")
def load_documents(filename="youtube_transcript.txt"):
    loader = TextLoader(filename, encoding="utf-8")
    return loader.load()

# ========================== Text Processing ==========================
@traceable(name="split_text")
def split_text(documents, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)

# ========================== Vector Store ==========================
@traceable(name="build_vectorstore")
def build_vectorstore(chunks):
    embeddings = init_embeddings()
    return FAISS.from_documents(chunks, embeddings)

@traceable(name="get_retriever")
def get_retriever(vectorstore, k=3):
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})

# ========================== Pipeline Setup ==========================
@traceable(name="youtube_setup_pipeline", tags=["setup"])
def setup_pipeline(youtube_url: str):
    """Parent setup traced function: builds retriever from a YouTube URL."""

    video_id = extract_youtube_id(youtube_url)
    if not video_id:
        raise ValueError("Invalid YouTube URL")

    # ✅ 1️⃣ Check if retriever already exists in cache
    if video_id in RETRIEVER_CACHE:
        return RETRIEVER_CACHE[video_id]

    # 🧠 2️⃣ First-time processing
    transcript = fetch_transcript(video_id)
    transcript_en = translate_to_english(transcript)

    filename = save_transcript_to_file(transcript_en, f"{video_id}.txt")
    documents = load_documents(filename)

    chunks = split_text(documents)
    vectorstore = build_vectorstore(chunks)
    retriever = get_retriever(vectorstore)

    # 📝 Cache the retriever for future queries
    RETRIEVER_CACHE[video_id] = retriever
    return retriever

# ========================== RAG Query ==========================
def build_rag_chain(retriever):
    llm = init_llm().with_config({"run_name": "llm"})

    prompt = PromptTemplate.from_template(
        """
        You are a helpful assistant.
        Answer ONLY from the provided transcript context.
        If the context is insufficient, just say you don't know.

        Context: {context}
        Question: {question}
        """
    ).with_config({"run_name": "prompt"})

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    retriever = retriever.with_config({"run_name": "retriever"})

    parallel = RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
    })

    return parallel | prompt | llm | StrOutputParser()

# ========================== Full Run ==========================
@traceable(name="youtube_rag_full_run")
def setup_pipeline_and_query(youtube_url: str, question: str):
    retriever = setup_pipeline(youtube_url)
    rag_chain = build_rag_chain(retriever)
    lc_config = {"run_name": "youtube_rag_query"}
    return rag_chain.invoke(question, config=lc_config)

# ========================== CLI / Test ==========================
if __name__ == "__main__":
    YT_URL = "https://www.youtube.com/watch?v=fZM3oX4xEyg"  # example

    while True:
        q = input("\nQ: ").strip()
        if not q:
            break
        answer = setup_pipeline_and_query(YT_URL, q)
        print("\nA:", answer)

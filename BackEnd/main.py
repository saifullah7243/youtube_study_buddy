# Import libraries
import os
import re
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langdetect import detect
from deep_translator import GoogleTranslator
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableParallel
from langchain.schema.output_parser import StrOutputParser
from youtube_transcript_api import YouTubeTranscriptApi
from dotenv import load_dotenv

load_dotenv()


# =========================  LLM & Embeddings  =========================
def init_llm(model_name="gpt-4.1-nano"):
    return init_chat_model(model_name, model_provider="openai")


def init_embeddings(model="text-embedding-3-small"):
    return OpenAIEmbeddings(model=model)


# =========================  YouTube Helpers  =========================
def extract_youtube_id(url: str) -> str | None:
    """Extract YouTube video ID from a URL."""
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})", url)
    return match.group(1) if match else None


def fetch_transcript(video_id: str, languages=None) -> str:
    """Fetch transcript text for a given YouTube video."""
    if languages is None:
        languages = ['en', 'hi', 'de', 'zh', 'en-US', 'fr', 'pa', 'ur', 'tr', 'te', 'es']

    transcript_list = YouTubeTranscriptApi().fetch(video_id, languages=languages)
    return " ".join(snippet.text for snippet in transcript_list)


# =========================  Translation  =========================
def translate_to_english(text: str, max_chunk_len=500) -> str:
    """Translate text to English if not already in English."""
    detected_lang = detect(text)
    if detected_lang == "en":
        return text

    chunks = [text[i:i + max_chunk_len] for i in range(0, len(text), max_chunk_len)]
    translated_chunks = [
        GoogleTranslator(source=detected_lang, target="en").translate(chunk)
        for chunk in chunks
    ]
    return " ".join(translated_chunks)

# =========================  File Helpers  =========================
def save_transcript_to_file(text:str, file_name="youtube_transcript.txt"):
    with open(file_name, "w", encoding="utf-8") as f:
        f.write(text)
    return file_name

## text loader
def load_documents(filename="youtube_transcript.txt"):
    loader = TextLoader(filename, encoding="utf-8")
    return loader.load()

# =========================  Text Processing  =========================
## splitiing text
def split_text(documents, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)

# =========================  Vector Store  =========================
## Embedding generation and storing
def build_vectorstore(chunks,embeddings):
    return FAISS.from_documents(chunks, embeddings)

## Retriever
def get_retriever(vectorstore, k=2):
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})

# =========================  RAG Chain =========================
## Generation
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
answer = rag_chain.invoke("What is this video about?")
print(answer)


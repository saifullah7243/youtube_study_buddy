from youtube_transcript_api import YouTubeTranscriptApi
import re
from langdetect import detect
from deep_translator import GoogleTranslator
from langsmith import traceable

# =========================  YouTube Helpers  =========================
@traceable(name="extract youtube id")
def extract_youtube_id(url: str) -> str | None:
    """Extract YouTube video ID from a URL."""
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})", url)
    return match.group(1) if match else None

@traceable(name="fetch youtube transcript from url")
def fetch_transcript(video_id: str, languages=None) -> str:
    """Fetch transcript text for a given YouTube video."""
    if languages is None:
        languages = ['en', 'hi', 'de', 'zh', 'en-US', 'fr', 'pa', 'ur', 'tr', 'te', 'es']

    transcript_list = YouTubeTranscriptApi().fetch(video_id, languages=languages)
    return " ".join(snippet.text for snippet in transcript_list)

@traceable(name="transcript translate to english")
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
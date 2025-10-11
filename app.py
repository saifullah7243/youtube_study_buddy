import streamlit as st
from BackEnd.main import setup_pipeline  # your retriever setup
from BackEnd.main import build_rag_chain  # to build query chain

st.set_page_config(page_title="🎓 YouTube Study Buddy", page_icon="🎥", layout="centered")

st.title("🎓 YouTube Study Buddy")
st.write("Ask questions about any YouTube video using AI.")

# Keep retriever & chat history in session
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_url" not in st.session_state:
    st.session_state.current_url = None

# Step 1: Get YouTube URL
youtube_url = st.text_input("Paste YouTube URL", placeholder="https://www.youtube.com/watch?v=...")

if st.button("Load Video"):
    if not youtube_url.strip():
        st.warning("Please enter a valid YouTube URL.")
    else:
        with st.spinner("⏳ Processing video... fetching transcript, embedding chunks..."):
            try:
                retriever = setup_pipeline(youtube_url)
                st.session_state.retriever = retriever
                st.session_state.current_url = youtube_url
                st.session_state.messages = []  # reset chat
                st.success("✅ Video processed successfully! You can start chatting below.")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# Step 2: Chat interface
if st.session_state.retriever:
    st.subheader("💬 Chat with the transcript")

    # Display previous messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # User input
    if prompt := st.chat_input("Ask something about the video..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    rag_chain = build_rag_chain(st.session_state.retriever)
                    answer = rag_chain.invoke(prompt)
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Error: {e}")

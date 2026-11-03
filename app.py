import streamlit as st
from langchain_ollama import ChatOllama

from src.ingestion import load_and_chunk_pdf
from src.vector_store import create_vector_store

st.set_page_config(page_title="Hybrid RAG Chatbot", page_icon="🤖")
st.title("🤖 Hybrid RAG Chatbot")
st.caption("Upload a PDF, index it, and ask questions.")

if "indexed" not in st.session_state:
    st.session_state.indexed = False
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chunks" not in st.session_state:
    st.session_state.chunks = []

uploaded_pdf = st.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_pdf is not None:
    st.info(f"Selected PDF: {uploaded_pdf.name}")

    if st.button("Index PDF", type="primary"):
        try:
            with st.spinner("Extracting PDF text..."):
                chunks = load_and_chunk_pdf(
                    uploaded_pdf,
                    filename=uploaded_pdf.name,
                )

            with st.spinner("Creating embeddings and indexing..."):
                vector_store = create_vector_store(chunks)

            st.session_state.chunks = chunks
            st.session_state.vector_store = vector_store
            st.session_state.indexed = True
            st.session_state.pdf_name = uploaded_pdf.name
            st.session_state.messages = []

            st.success(
                f"Indexed {len(chunks)} chunks. Your chatbot is ready!"
            )

        except Exception as exc:
            st.error(f"Indexing failed: {exc}")
            st.exception(exc)

if st.session_state.indexed:
    st.divider()
    st.subheader(f"💬 Chat with {st.session_state.pdf_name}")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask a question about your PDF...")

    if question:
        st.session_state.messages.append(
            {"role": "user", "content": question}
        )

        with st.chat_message("user"):
            st.markdown(question)

        try:
            with st.spinner("Searching your PDF and generating an answer..."):
                docs = st.session_state.vector_store.similarity_search(
                    question,
                    k=4,
                )

                if not docs:
                    answer = "I couldn't find relevant information in the PDF."
                else:
                    context = "\n\n".join(
                        f"Source: {doc.metadata.get('source', 'PDF')}, "
                        f"page {doc.metadata.get('page', '?')}\n"
                        f"{doc.page_content}"
                        for doc in docs
                    )

                    llm = ChatOllama(
                        model="llama3.2:latest",
                        temperature=0,
                    )

                    response = llm.invoke(
                        "Answer the question using only the provided context. "
                        "If the context does not contain the answer, say so. "
                        "Mention relevant source page numbers.\n\n"
                        f"CONTEXT:\n{context}\n\n"
                        f"QUESTION:\n{question}"
                    )
                    answer = response.content

            with st.chat_message("assistant"):
                st.markdown(answer)

            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )

        except Exception as exc:
            st.error(f"Answer generation failed: {exc}")
            st.exception(exc)

import streamlit as st
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import os
import warnings

# -------------------- App Config --------------------
st.set_page_config(page_title="PDF Chatbot", page_icon="📚")

# -------------------- Session State --------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -------------------- Embedding Model --------------------
@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # or L12-v2

# -------------------- LLM --------------------
@st.cache_resource
def load_llm():
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=512)
    return HuggingFacePipeline(pipeline=pipe)

# -------------------- PDF Processing --------------------
def extract_pdf_text(uploaded_file):
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        return [page.get_text() for page in doc]

# -------------------- Chunking --------------------
def split_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    return splitter.split_text(text)

# -------------------- RAG Chain Setup --------------------
def setup_qa_chain(chunks):
    embeddings = load_embedding_model()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vectorstore = FAISS.from_texts(chunks, embedding=embeddings)

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    llm = load_llm()

    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are a helpful assistant. Answer the user's question based on the context from a PDF document.

Your tone should be clear, friendly, and conversational — do not copy the style of the original document. However, you must only use the information found in the context. If the answer isn't in the context, say so clearly.

Context:
{context}

Question:
{question}

Answer:
"""
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt_template},
        return_source_documents=True
    )

    return chain

# -------------------- UI Elements --------------------
st.title("📚 PDF Chatbot")
st.markdown("Upload a PDF and ask questions about its content")

# Sidebar
with st.sidebar:
    st.header("Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    st.markdown("---")
    st.markdown("ℹ️ Uses FLAN-T5-Base + MiniLM-L6-v2 embeddings")

# -------------------- Chat Section --------------------
if uploaded_file:
    pages_text = extract_pdf_text(uploaded_file)
    full_text = " ".join(pages_text)

    if not full_text.strip():
        st.error("The uploaded PDF appears to be empty or has no extractable text.")
    else:
        chunks = split_text(full_text)
        qa_chain = setup_qa_chain(chunks)

        st.subheader("Chat")
        for qa in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(qa["question"])
            with st.chat_message("assistant"):
                st.write(qa["answer"])

        question = st.chat_input("Ask a question about the PDF...")

        if question:
            with st.chat_message("user"):
                st.write(question)

            with st.spinner("Thinking..."):
                result = qa_chain({"query": question})
                answer = result["result"]
                sources = result.get("source_documents", [])

            with st.chat_message("assistant"):
                st.write(answer)
                if sources:
                    with st.expander("📄 Source Documents"):
                        for i, doc in enumerate(sources):
                            st.markdown(f"**Source {i+1}:**")
                            st.code(doc.page_content.strip(), language="markdown")

            st.session_state.chat_history.append({
                "question": question,
                "answer": answer
            })
else:
    st.info("Please upload a PDF file to get started.")

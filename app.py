import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. Setup & Config
load_dotenv()
st.set_page_config(page_title="Document Q&A Bot", layout="centered")
st.title("🤖 Document Q&A Assistant")

# 2. Initialize Components (Cached so they don't reload every click)
@st.cache_resource
def init_rag():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # Load existing database
    vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    
    system_prompt = (
        "You are a strict technical assistant. Use ONLY the provided context to answer. "
        "If the specific topic or entity (e.g., a specific app or brand name) mentioned in the "
        "question is not explicitly discussed in the context, you must state: "
        "'I'm sorry, but the provided documents do not contain information about [Topic].' "
        "Do not attempt to find similar-sounding words or concepts. "
        "Do not use external knowledge. \n\n"
        "Context: {context}"
    )
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
    
    
    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 3})
    qa_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, qa_chain)

rag_chain = init_rag()

# 3. User Interface
user_query = st.text_input("Ask a question about your documents:")

if user_query:
    with st.spinner("Searching documents..."):
        response = rag_chain.invoke({"input": user_query})
        output = response["answer"]

        # Display Answer
        st.subheader("Answer")
        st.write(output)

        negative_phrases = ["i'm sorry", "don't know", "not mentioned", "not contained"]

        is_not_found = any(phrase in output.lower() for phrase in negative_phrases)

        if not is_not_found:
            st.subheader("Source Citations")
            # We use a set to ensure unique filenames
            sources = {os.path.basename(doc.metadata['source']) for doc in response['context']}

            for s in sources:
                st.info(f"📄 Source: {os.path.basename(s)}")
        
            with st.expander("View Retrieved Text Chunks"):
                for i, doc in enumerate(response['context']):
                    st.write(f"**Chunk {i+1} from {os.path.basename(doc.metadata['source'])}:**")
                    st.write(f"{doc.page_content[:300]}...")
        else:
            # If the answer was "I don't know", we explain why no sources are shown
            st.warning("No relevant sources found in the provided documents for this query.")
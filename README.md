# Multi-Document Technical Q&A Bot (RAG)

## 1. Project Description
This project is an interactive Retrieval-Augmented Generation (RAG) system designed to answer technical questions based on a private collection of PDFs (SQL, Python, and AI/ML notes). The bot uses semantic search to find the most relevant document snippets and leverages an LLM to provide grounded, accurate answers with clear source citations.

## 2. Tech Stack
| Tool / Library | Version | Purpose |
| :--- | :--- | :--- |
| **Python** | 3.11+ | Primary Programming Language |
| **Streamlit** | 1.32.0 | Web User Interface |
| **LangChain-Classic** | 1.2.13 | RAG Orchestration and Chains |
| **Google Generative AI** | 0.4.1 | LLM (Gemini 1.5 Flash) |
| **ChromaDB** | 0.4.24 | Persistent Vector Database |
| **HuggingFace** | 2.2.4 | Local Embedding Generation |
| **PyPDF** | 4.1.0 | PDF Text Extraction |
| **Python-Dotenv** | 1.0.1 | Environment Variable Management |

## 3. Architecture Overview
The system follows a standard RAG pipeline:
1.  **Ingestion:** PDFs are loaded from the `/data` folder using `PyPDFLoader`.
2.  **Chunking:** Text is split into smaller segments for better retrieval granularity.
3.  **Embedding:** Chunks are converted into numerical vectors using a local HuggingFace model.
4.  **Retrieval:** When a user asks a question, the system finds the top-k most similar chunks in ChromaDB using **Maximum Marginal Relevance (MMR)** to ensure diverse context.
5.  **Generation:** The LLM receives the context and the query to generate a response strictly grounded in the provided documents.

## 4. Chunking Strategy
- **Strategy:** `RecursiveCharacterTextSplitter`
- **Parameters:** Chunk Size: 1000 characters | Chunk Overlap: 200 characters.
- **Reasoning:** 1000 characters is ideal for technical documentation to ensure a "definition" or "code block" isn't cut off mid-sentence. The 200-character overlap maintains context between chunks, ensuring that relationships between concepts are preserved at the split points.

## 5. Embedding Model & Vector Database
- **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` (via HuggingFace).
    - **Why:** It provides high-performance semantic search locally, avoiding the latency and costs associated with cloud-based embedding APIs.
- **Vector Database:** `ChromaDB`.
    - **Why:** It is lightweight, supports native persistence for local vector storage, and allows for efficient metadata filtering for source citations.

## 6. Setup Instructions
1.  **Clone the Repo:**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-directory>
    ```
2.  **Create a Virtual Environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Run the App:**
    ```bash
    streamlit run app.py
    ```

## 7. Environment Variables
To run this project, you must create a `.env` file in the root directory and add your Google API Key:
```text
GOOGLE_API_KEY=your_actual_api_key_here

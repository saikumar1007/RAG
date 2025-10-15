# RAG
Conversational AI for PDF Q&amp;A using Retrieval-Augmented Generation (RAG)


Absolutely! Here’s a professional **README.md** you can use for your **RAG** project. It’s structured to impress a CEO, investor, or tech reviewer:

````markdown
# RAG: Conversational AI for PDF Q&A

**RAG** is a powerful AI assistant that lets you **interactively ask questions from your PDF documents** using **Retrieval-Augmented Generation (RAG)**. The system provides precise, concise answers **extracted directly from your uploaded PDFs**, making it ideal for research, business reports, or presentations.

---

## 🚀 Features

- **PDF-based Q&A**: Ask questions and get answers strictly from your uploaded documents.  
- **Retrieval-Augmented Generation (RAG)**: Combines embeddings and LLMs for accurate context-aware responses.  
- **Conversational History**: Maintains chat history for context-aware question reformulation.  
- **Secure Deployment**: API keys are hidden; compatible with Streamlit Cloud or local deployment.  
- **Professional Interface**: Modern, scrollable chat UI optimized for clarity and readability.  

---

## 📄 How It Works

1. Upload one or more PDF documents.  
2. The system **splits your PDFs into chunks** and generates embeddings for retrieval.  
3. Ask a question in the chat input.  
4. The assistant **retrieves relevant chunks** and provides a concise answer.  
5. Chat history is maintained for context-aware follow-up questions.  

---

## ⚡ Tech Stack

- **Language Model**: Groq LLM (Llama 3.1 8B)  
- **Embeddings**: HuggingFace `all-MiniLM-L6-v2`  
- **Vector Store**: Chroma  
- **Framework**: Streamlit for interactive UI  
- **Document Loader**: PyPDFLoader  
- **Prompt Management**: LangChain  

---

## 🛠 Setup Instructions

1. Clone the repository:  
```bash
git clone https://github.com/<your-username>/RAG.git
cd RAG
````

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set environment variables (locally or via Streamlit Secrets):

```bash
GROQ_API_KEY="your_groq_api_key_here"
HF_TOKEN="your_huggingface_token_here"
```

4. Run the app locally:

```bash
streamlit run app.py
```



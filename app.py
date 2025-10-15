import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os
from dotenv import load_dotenv

# Load local environment variables (ignored by GitHub)
load_dotenv()

# Fetch API keys from environment (or Streamlit Secrets)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

os.environ['HF_TOKEN'] = HF_TOKEN

# Embeddings model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# --- Streamlit Layout ---
st.set_page_config(
    page_title="AI PDF Insight Explorer",
    layout="wide",
    page_icon="📄"
)

# App Title & Description
st.title("📄 AI PDF Insight Explorer")
st.markdown("""
**Welcome to AI PDF Insight Explorer** – a professional, intelligent PDF Q&A assistant.  

Upload your PDF documents and ask questions. The assistant provides **concise, precise answers extracted only from your documents**.  

Perfect for research, business reports, or presentations.
""")
st.markdown("---")

# Initialize LLM silently
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.1-8b-instant")

# Stateful chat management
if 'store' not in st.session_state:
    st.session_state.store = {}
if 'chat_history_ui' not in st.session_state:
    st.session_state.chat_history_ui = []

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

# --- File Upload ---
uploaded_files = st.file_uploader(
    "Choose PDF files", type="pdf", accept_multiple_files=True, label_visibility="collapsed"
)

documents = []
if uploaded_files:
    for uploaded_file in uploaded_files:
        temp_pdf = f"./temp_{uploaded_file.name}"
        with open(temp_pdf, "wb") as file:
            file.write(uploaded_file.getvalue())
        loader = PyPDFLoader(temp_pdf)
        docs = loader.load()
        documents.extend(docs)

if documents:
    # Split and embed
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    splits = text_splitter.split_documents(documents)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    # History-aware reformulation
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given chat history and a user question, reformulate the question to be standalone. Do NOT answer."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    # QA chain (ONLY PDF context)
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Answer the user question using ONLY the provided PDF context. "
         "If the answer is not in the PDFs, reply: 'The answer is not available in the uploaded documents.' "
         "Keep it concise, max 3 sentences.\n\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    # --- Chat Input & History ---
    st.markdown("### Ask a question from your PDFs:")
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input("", placeholder="Type your question here...")
        submitted = st.form_submit_button("Send")
        if submitted and user_input:
            session_history = get_session_history("default_session")
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": "default_session"}}
            )
            # Only store/display PDF-based answer
            st.session_state.chat_history_ui.append({
                "user": user_input,
                "assistant": response['answer']
            })

    # --- Display chat history ---
    st.markdown("<div style='max-height:500px; overflow-y:auto; padding:10px; background:#f7f7f7; border-radius:10px;'>", unsafe_allow_html=True)
    for chat in st.session_state.chat_history_ui:
        # User message
        st.markdown(f"""
        <div style='background:#DCF8C6; padding:8px; border-radius:10px; margin-bottom:5px; max-width:70%; align-self:flex-end;'>
        {chat['user']}
        </div>""", unsafe_allow_html=True)
        # Assistant message
        st.markdown(f"""
        <div style='background:#FFFFFF; padding:8px; border-radius:10px; margin-bottom:5px; max-width:70%; align-self:flex-start;'>
        {chat['assistant']}
        </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

"""
Smart Resume Matcher & AI Chatbot
A beautiful, modern Streamlit application with:
- Resume matching using SBERT similarity
- RAG-based AI chatbot for resume/job questions
"""

import os
import time
from datetime import datetime

import docx
import fitz
import pandas as pd
import requests
import streamlit as st
import base64

# ========================================
# CONFIGURATION
# ========================================
API_BASE = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
MATCH_ENDPOINT = f"{API_BASE}/match_resume"
CHAT_ENDPOINT = f"{API_BASE}/ask"

# ========================================
# CUSTOM CSS - Modern Dark Glassmorphism Theme
# ========================================
def inject_custom_css():
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        background-attachment: fixed;
    }
    
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: rgba(15, 12, 41, 0.95);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    [data-testid="stSidebar"] .css-1d391kg {
        padding-top: 2rem;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-weight: 600;
    }
    
    /* Text */
    p, span, label, .stMarkdown {
        color: #e0e0e0;
    }
    
    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        background: rgba(255, 255, 255, 0.08);
        border-color: rgba(138, 43, 226, 0.5);
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(138, 43, 226, 0.2);
    }
    

    
    /* Input Fields */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 12px !important;
        color: white !important;
        padding: 12px 16px !important;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2) !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        text-transform: none !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4) !important;
    }
    
    /* File Uploader */
    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.05);
        border: 2px dashed rgba(255, 255, 255, 0.2);
        border-radius: 16px;
        padding: 1rem;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #667eea;
    }
    
    /* Progress Bars */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
        border-radius: 10px;
    }
    
    .stProgress > div > div {
        background: rgba(255, 255, 255, 0.1) !important;
        border-radius: 10px;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.05) !important;
        border-radius: 12px !important;
        color: white !important;
    }
    
    .streamlit-expanderContent {
        background: rgba(255, 255, 255, 0.02) !important;
        border-radius: 0 0 12px 12px !important;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #667eea !important;
    }
    
    /* Alerts/Info boxes */
    .stAlert {
        background: rgba(255, 255, 255, 0.05) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    
    /* Navigation Pills */
    .nav-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.75rem 1.5rem;
        border-radius: 50px;
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: #e0e0e0;
        cursor: pointer;
        transition: all 0.3s ease;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
    }
    
    .nav-pill:hover, .nav-pill.active {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-color: transparent;
        color: white;
    }
    
    /* Score Badge */
    .score-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.5rem 1rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    .score-high {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
    }
    
    .score-medium {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    
    .score-low {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
    }
    
    /* Logo/Brand */
    .brand-logo {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 2rem;
    }
    
    /* Metric Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .metric-label {
        color: #a0a0a0;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.05);
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.2);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(255, 255, 255, 0.3);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 12px;
        color: #e0e0e0;
        padding: 0.75rem 1.5rem;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
    }
    
    /* Chat container */
    .chat-container {
        height: 500px;
        overflow-y: auto;
        padding: 1rem;
        background: rgba(0, 0, 0, 0.2);
        border-radius: 16px;
        margin-bottom: 1rem;
    }
    
    /* Guardrails info */
    .guardrails-info {
        background: rgba(102, 126, 234, 0.1);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 12px;
        padding: 1rem;
        margin-top: 0.5rem;
        font-size: 0.85rem;
        color: #a0a0a0;
    }
    
    /* Typing indicator */
    .typing-indicator {
        display: flex;
        gap: 4px;
        padding: 1rem;
    }
    
    .typing-dot {
        width: 8px;
        height: 8px;
        background: #667eea;
        border-radius: 50%;
        animation: typing 1.4s infinite ease-in-out;
    }
    
    .typing-dot:nth-child(1) { animation-delay: 0s; }
    .typing-dot:nth-child(2) { animation-delay: 0.2s; }
    .typing-dot:nth-child(3) { animation-delay: 0.4s; }
    
    @keyframes typing {
        0%, 100% { transform: scale(1); opacity: 0.5; }
        50% { transform: scale(1.2); opacity: 1; }
    }
    </style>
    """, unsafe_allow_html=True)


# ========================================
# HELPER FUNCTIONS
# ========================================
def extract_text_from_file(uploaded_file):
    """Extract text from uploaded resume file"""
    try:
        if uploaded_file.name.endswith(".pdf"):
            doc = fitz.open(stream=uploaded_file.getvalue(), filetype="pdf")
            return "".join(page.get_text() for page in doc)
        elif uploaded_file.name.endswith(".docx"):
            doc = docx.Document(uploaded_file)
            return "\n".join([p.text for p in doc.paragraphs])
        elif uploaded_file.name.endswith(".txt"):
            return uploaded_file.getvalue().decode("utf-8")
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None


def analyze_resume_with_retry(payload, max_retries=3, delay=2, timeout=30):
    """Call the resume matching API with retry logic"""
    for attempt in range(max_retries):
        try:
            resp = requests.post(MATCH_ENDPOINT, json=payload, timeout=timeout)
            if resp.status_code == 422:
                return False, None, f"Validation error: {resp.text}"
            resp.raise_for_status()
            return True, resp.json(), None
        except requests.exceptions.RequestException as e:
            if attempt >= max_retries - 1:
                return False, None, f"Backend not responding ({e})"
            time.sleep(delay)
    return False, None, "Backend service not responding"


def send_chat_message(question, user_context=None, timeout=60):
    """Send a message to the RAG chatbot API"""
    try:
        payload = {"question": question, "user_context": user_context}
        resp = requests.post(CHAT_ENDPOINT, json=payload, timeout=timeout)
        if resp.status_code == 400:
            error_detail = resp.json().get("detail", {})
            return False, None, f"Input validation failed: {error_detail}"
        resp.raise_for_status()
        return True, resp.json(), None
    except requests.exceptions.RequestException as e:
        return False, None, f"Chat service unavailable: {e}"


def get_score_class(score):
    """Return CSS class based on score value"""
    if score >= 0.7:
        return "score-high"
    elif score >= 0.4:
        return "score-medium"
    return "score-low"


def load_job_descriptions():
    """Load job descriptions from CSV file"""
    job_file_paths = [
        "job_title_des.csv",
        "notebooks/job_title_des.csv",
        "src/ui/notebooks/job_title_des.csv",
    ]
    
    df_jobs = None
    for path in job_file_paths:
        try:
            if os.path.exists(path):
                df_jobs = pd.read_csv(path)
                break
        except Exception:
            continue
    
    return df_jobs


# ========================================
# PAGE COMPONENTS
# ========================================
def render_header():
    """Render the main header"""
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 style="font-size: 2.5rem; margin-bottom: 0.5rem;">
            ✨ Resume Matcher & AI Assistant
        </h1>
        <p style="color: #a0a0a0; font-size: 1.1rem;">
            Upload your resume, find matching jobs, and chat with our AI about career advice
        </p>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Render the sidebar navigation"""
    with st.sidebar:
        st.markdown('<div class="brand-logo">🚀 ResumeAI</div>', unsafe_allow_html=True)
        
        # Navigation
        st.markdown("### Navigation")
        page = st.radio(
            "Choose a feature:",
            ["📄 Resume Matcher", "💬 AI Chatbot", "ℹ️ About"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Resume Upload (shared between features)
        st.markdown("### 📎 Your Resume")
        uploaded_file = st.file_uploader(
            "Upload resume (PDF, DOCX, TXT)",
            type=["txt", "pdf", "docx"],
            key="resume_upload"
        )
        
        if uploaded_file:
            resume_text = extract_text_from_file(uploaded_file)
            if resume_text:
                st.session_state.resume_text = resume_text
                st.success(f"✅ Loaded: {uploaded_file.name}")
                with st.expander("Preview"):
                    st.text(resume_text[:500] + "..." if len(resume_text) > 500 else resume_text)
        
        st.markdown("---")
        
        # Status indicators
        st.markdown("### 🔌 System Status")
        
        # Check API health
        try:
            health = requests.get(f"{API_BASE}/health", timeout=5)
            if health.status_code == 200:
                st.success("API: Connected")
            else:
                st.warning("API: Degraded")
        except:
            st.error("API: Offline")
        
        return page


def render_resume_matcher():
    """Render the Resume Matcher page"""
    
    # Hero / Upload Section
    st.markdown("## 🎯 Find Your Perfect Job Match")
    st.markdown("Upload your resume and discover the best matching job opportunities.")

    with st.container():
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 📤 Upload Resume")
        uploaded_file = st.file_uploader(
            "Upload your resume (PDF, DOCX, TXT)",
            type=["txt", "pdf", "docx"],
            key="resume_upload_main",
            label_visibility="collapsed"
        )
        
        if uploaded_file:
            resume_text = extract_text_from_file(uploaded_file)
            if resume_text:
                st.session_state.resume_text = resume_text
                st.success(f"✅ Loaded: {uploaded_file.name}")
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Load job descriptions
    df_jobs = load_job_descriptions()
    
    if df_jobs is None:
        st.error("❌ Job database not found. Please ensure job_title_des.csv exists.")
        return
    
    try:
        job_descriptions_dict = dict(zip(df_jobs["Job Title"], df_jobs["Job Description"]))
    except KeyError as e:
         st.error(f"CSV file missing required column: {e}")
         return
    
    # Input & Settings
    with st.container():
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        col1, col2 = st.columns([3, 1])
        
        with col1:
             st.markdown("### 📝 Resume Content")
             resume_text = st.session_state.get("resume_text", "")
             resume_input = st.text_area(
                "Verify or edit your resume text:",
                value=resume_text,
                height=200,
                placeholder="Content will appear here after upload...",
                key="resume_matcher_input",
                label_visibility="collapsed"
            )

        with col2:
            st.markdown("### ⚙️ Filters")
            top_n = st.slider("Max Matches", 1, 20, 5)
            st.caption(f"Searching {len(df_jobs)} jobs")
            
            st.markdown("---")
            analyze_btn = st.button("🚀 Analyze", use_container_width=True, type="primary")

        st.markdown('</div>', unsafe_allow_html=True)

    # Analysis & Results
    if analyze_btn:
        effective_resume = resume_input or resume_text
        
        if not effective_resume.strip():
            st.warning("⚠️ Please provide your resume content first!")
            return

        with st.spinner("🔍 Analyzing resume against job database..."):
            payload = {"resume_text": effective_resume, "top_n": top_n}
            success, result, error_msg = analyze_resume_with_retry(payload)
            
            if success:
                st.toast("Analysis complete!", icon="✅")
                matches = result["matches"]
                
                st.markdown("### 📋 Top Matching Jobs")
                
                for match in matches:
                    score = match["similarity_score"]
                    score_class = get_score_class(score)
                    job_title = match["job_title"]
                    desc = job_descriptions_dict.get(job_title, "No description available.")
                    
                    # Enhanced Card Layout
                    st.markdown(f"""
                    <div class="glass-card" style="border-left: 5px solid {
                        '#38ef7d' if score >= 0.7 else '#f5576c' if score >= 0.4 else '#4facfe'
                    };">
                        <div style="display: flex; justify-content: space-between; align-items: start;">
                            <div>
                                <h3 style="margin: 0; color: white;">{match["rank"]}. {job_title}</h3>
                                <p style="margin: 5px 0 0 0; font-size: 0.9em; opacity: 0.8;">
                                    {desc[:150]}...
                                </p>
                            </div>
                            <div style="text-align: right; min-width: 100px;">
                                <span class="score-badge {score_class}" style="font-size: 1.1em;">
                                    {score:.0%} Match
                                </span>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Expandable details
                    with st.expander(f"View Details for {job_title}"):
                        st.markdown(f"**Relevance Score:** {score:.4f}")
                        st.markdown("#### Job Description")
                        st.write(desc)
                        st.button(f"Chat about this job", key=f"chat_{match['rank']}", 
                                  help="Ask the AI assistant specifically about this role (Coming Soon)")
                    
            else:
                st.error(f"❌ {error_msg}")


def render_chatbot():
    """Render the AI Chatbot page"""
    st.markdown("## 💬 AI Career Assistant")
    
    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # --- 1. RENDER HISTORY ---
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            # Show guardrails info if available
            if "guardrails" in msg and msg["guardrails"]:
                guardrails = msg["guardrails"]
                input_validation = guardrails.get('input_validation') or {}
                risk_level = input_validation.get('risk_level', 'N/A')
                
                if risk_level != 'N/A':
                    st.caption(f"🛡️ Guardrails: Risk level: {risk_level}")
            
            # Show download button if PDF was generated
            if "generated_pdf" in msg and msg["generated_pdf"]:
                pdf_data = base64.b64decode(msg["generated_pdf"])
                st.download_button(
                    label="📄 Download Edited Resume (PDF)",
                    data=pdf_data,
                    file_name="Edited_Resume.pdf",
                    mime="application/pdf",
                    key=f"dl_{st.session_state.chat_history.index(msg)}"
                )

    # --- 2. GENERATE RESPONSE ---
    # Check if the last message is from the user (implies we need to respond)
    if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Get context
                resume_context = st.session_state.get("resume_text", "")
                
                last_msg_obj = st.session_state.chat_history[-1]
                # Use hidden_content (with attachment) if available, else displayed content
                query_text = last_msg_obj.get("hidden_content", last_msg_obj["content"])
                
                # Call API
                success, response, error = send_chat_message(query_text, user_context=resume_context)
                
                if success:
                    answer = response.get("answer", "I couldn't generate a response.")
                    st.markdown(answer)
                    
                    bot_message = {
                        "role": "assistant",
                        "content": answer,
                        "timestamp": datetime.now().isoformat(),
                        "guardrails": response.get("guardrails", {}),
                        "context": response.get("context_used", []),
                        "generated_pdf": response.get("generated_pdf")
                    }
                else:
                    error_msg = f"I'm sorry, I encountered an error: {error}. Please make sure the backend service is running."
                    st.error(error_msg)
                    bot_message = {
                        "role": "assistant",
                        "content": error_msg,
                        "timestamp": datetime.now().isoformat()
                    }
                
                # Add to history
                st.session_state.chat_history.append(bot_message)
                
        # Rerun to update the UI with the final message
        st.rerun()

    # --- 3. HANDLE INPUT ---
    # File Attachment Popover (Rendered below history, effectively at bottom of flow)
    attachment_text = ""
    with st.popover("📎 Attach Document", use_container_width=False):
        st.markdown("### Upload Document")
        chat_file = st.file_uploader("Upload PDF/DOCX/TXT", type=["pdf", "docx", "txt"], key="chat_attachment")
        if chat_file:
            with st.spinner("Processing attachment..."):
                extracted = extract_text_from_file(chat_file)
                if extracted:
                    attachment_text = f"\n\n[Attached Document Context]:\n{extracted}\n"
                    st.success("✅ File attached! It will be sent with your next message.")
                else:
                    st.error("❌ Failed to read file.")

    # Chat Input (Always visible, pinned to bottom)
    user_input = st.chat_input("Ask me about job requirements, resume tips, career advice...")
    
    # Handle Quick Questions (only if history is empty)
    selected_question = None
    if not st.session_state.chat_history:
        st.markdown("### 💡 Quick Questions")
        cols = st.columns(3)
        questions = [
            "What skills do Data Scientists need?",
            "How to improve my resume?",
            "Common interview questions"
        ]
        for i, q in enumerate(questions):
            if cols[i].button(q, use_container_width=True):
                selected_question = q

    # Determine final prompt (Input Bar takes precedence, but usually mutually exclusive in usage)
    prompt = user_input or selected_question

    if prompt:
        # Append attachment text to the prompt internally (hidden from UI to keep chat clean?) 
        # Or just send it. Let's send it as part of context.
        # Ideally we want to see "User attached: filename" in chat.
        
        display_prompt = prompt
        full_content = prompt + attachment_text
        
        # If attachment exists, add a note to display
        if attachment_text and chat_file:
             display_prompt += f" \n\n*📎 Attached: {chat_file.name}*"

        st.session_state.chat_history.append({
            "role": "user",
            "content": display_prompt, 
            "hidden_content": full_content, # Store full content for API processing
            "timestamp": datetime.now().isoformat()
        })
        st.rerun()

    # Clear chat button
    if st.session_state.chat_history:
        st.markdown("---")
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()


def render_about():
    """Render the About page"""
    st.markdown("## ℹ️ About Resume Matcher & AI Assistant")
    
    st.markdown("""
    <div class="glass-card">
        <h3>🚀 What is this?</h3>
        <p>This is a powerful AI-driven platform that helps job seekers:</p>
        <ul>
            <li><strong>Match resumes to jobs</strong> using advanced SBERT embeddings</li>
            <li><strong>Get career advice</strong> through our RAG-based AI chatbot</li>
            <li><strong>Understand job requirements</strong> better with AI explanations</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">SBERT</div>
            <div class="metric-label">Embedding Model</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">Llama 3.3</div>
            <div class="metric-label">LLM Backend</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">FAISS</div>
            <div class="metric-label">Vector Store</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("")
    
    st.markdown("""
    <div class="glass-card">
        <h3>🛡️ Safety Features</h3>
        <p>Our platform includes comprehensive guardrails:</p>
        <ul>
            <li><strong>PII Detection</strong> - Protects sensitive information</li>
            <li><strong>Injection Filtering</strong> - Prevents prompt attacks</li>
            <li><strong>Toxicity Filtering</strong> - Ensures appropriate responses</li>
            <li><strong>Hallucination Detection</strong> - Validates AI responses</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="glass-card">
        <h3>👥 Team</h3>
        <p>Built with ❤️ as part of an MLOps group project.</p>
    </div>
    """, unsafe_allow_html=True)


# ========================================
# MAIN APPLICATION
# ========================================
def main():
    # Page configuration
    st.set_page_config(
        page_title="Resume Matcher & AI Assistant",
        page_icon="✨",
        layout="wide",
        initial_sidebar_state="collapsed" 
    )
    
    # Initialize session state
    if "resume_text" not in st.session_state:
        st.session_state.resume_text = ""
    
    # Inject custom CSS
    inject_custom_css()
    
    # Render header
    render_header()
    
    # Top Level Navigation
    tab1, tab2, tab3 = st.tabs(["📄 Resume Matcher", "💬 AI Chatbot", "ℹ️ About"])
    
    with tab1:
        render_resume_matcher()
    
    with tab2:
        render_chatbot()
        
    with tab3:
        render_about()

if __name__ == "__main__":
    main()

"""
Streamlit Chat Interface for Restaurant Recommendation System
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title=" AI Restaurant Discovery",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
# Custom CSS
st.markdown("""
<style>
    /* Main container background */
    .stApp {
        background-color: #1e1e1e;
    }
    
    /* Chat messages */
    .stChatMessage {
        background-color: #2d2d2d !important;
        color: #ffffff !important;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    
    /* User message */
    [data-testid="stChatMessageContent"] {
        color: #ffffff !important;
    }
    
    /* Markdown text */
    .stMarkdown {
        color: #ffffff !important;
    }
    
    /* Chat input */
    .stChatInput {
        background-color: #2d2d2d !important;
        color: #ffffff !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #252525 !important;
    }
    
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
    }
    
    /* Buttons */
    .stButton button {
        background-color: #404040 !important;
        color: #ffffff !important;
        border: 1px solid #555555 !important;
    }
    
    .stButton button:hover {
        background-color: #505050 !important;
        border-color: #666666 !important;
    }
    
    /* Code blocks */
    code {
        background-color: #2d2d2d !important;
        color: #ff6b6b !important;
    }
    
    /* Links */
    a {
        color: #64b5f6 !important;
    }
    
    /* Input fields */
    input {
        background-color: #2d2d2d !important;
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)



def initialize_session():
    """Initialize session state variables."""
    if 'agent' not in st.session_state:
        st.session_state.agent = None
        st.session_state.initialized = False
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'processing' not in st.session_state:
        st.session_state.processing = False


def load_agent():
    """Load the conversational orchestrator."""
    if not st.session_state.initialized:
        with st.spinner("🚀 Initializing AI agents... (This may take a moment)"):
            try:
                # Use new 4-agent orchestrator
                from agents.orchestrator import ConversationalOrchestrator
                
                agent = ConversationalOrchestrator()
                agent.initialize()
                
                st.session_state.agent = agent
                st.session_state.initialized = True
                st.success("✅ AI agents ready!")
                return True
            except Exception as e:
                st.error(f"❌ Failed to initialize: {e}")
                st.info("Please ensure all dependencies are installed and data pipelines have been run.")
                return False
    return True


def display_sidebar():
    """Display sidebar with info and controls."""
    with st.sidebar:
        st.markdown("##  AI Restaurant Discovery")
        st.markdown("---")
        
        st.markdown("### About")
        st.markdown("""
        This AI-powered system helps you discover the perfect restaurant in Bangalore using:
        
        -  **Natural Language Understanding**
        -  **Semantic Search**
        -  **Smart Ranking**
        -  **Conversational AI**
        """)
        
        st.markdown("---")
        st.markdown("### Sample Queries")
        
        sample_queries = [
            "Romantic Italian restaurants in Koramangala",
            "Budget-friendly North Indian under ₹800",
            "Best cafes with good ambiance",
            "Anniversary dinner with table booking",
            "Highly rated Chinese food delivery"
        ]
        
        for query in sample_queries:
            if st.button(f" {query[:40]}...", key=f"sample_{query[:10]}"):
                st.session_state.sample_query = query
        
        st.markdown("---")
        st.markdown("### Session Info")
        
        if st.session_state.initialized and st.session_state.agent:
            state = st.session_state.agent.get_session_stats()
            st.markdown(f"**Session ID**: `{state.get('session_id', 'N/A')[:15]}...`")
            st.markdown(f"**Turns**: {state.get('turn_count', 0)}")
        
        if st.button(" Clear Conversation"):
            st.session_state.messages = []
            if st.session_state.agent:
                st.session_state.agent.session.conversation_history = []
                st.session_state.agent.session.last_results = []
            st.rerun()


def display_chat_history():
    """Display chat message history."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def process_query(query: str):
    """Process user query and get response."""
    try:
        response = st.session_state.agent.process(query)
        return response
    except Exception as e:
        return f" Error processing query: {str(e)}\n\nPlease try rephrasing your question."


def main():
    """Main application."""
    initialize_session()
    
    # Header
    st.markdown('<div class="main-header"> AI Restaurant Discovery</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Find the perfect restaurant in Bangalore with intelligent recommendations</div>', unsafe_allow_html=True)
    
    # Sidebar
    display_sidebar()
    
    # Load agent
    if not load_agent():
        st.stop()
    
    # Display chat history
    display_chat_history()
    
    # Handle sample query from sidebar
    if hasattr(st.session_state, 'sample_query') and st.session_state.sample_query:
        query = st.session_state.sample_query
        st.session_state.sample_query = None  # Clear
        
        # Add user message
        st.session_state.messages.append({"role": "user", "content": query})
        
        # Process and add response
        with st.spinner(" Thinking..."):
            response = process_query(query)
            st.session_state.messages.append({"role": "assistant", "content": response})
        
        st.rerun()
    
    # Chat input
    if prompt := st.chat_input("Ask me about restaurants in Bangalore..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Process and display response
        with st.chat_message("assistant"):
            with st.spinner(" Searching for the best restaurants..."):
                response = process_query(prompt)
                st.markdown(response)
        
        # Add response to history
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Welcome message for new sessions
    if not st.session_state.messages:
        with st.chat_message("assistant"):
            st.markdown("""
             **Welcome to AI Restaurant Discovery!**
            
            I can help you find the perfect restaurant in Bangalore. Try asking me:
            
            - "Find romantic restaurants for anniversary dinner"
            - "Best biryani places in Koramangala under ₹1000"
            - "Show me cafes with good ambiance"
            - "Highly rated North Indian restaurants"
            
            What kind of dining experience are you looking for today?
            """)


if __name__ == "__main__":
    main()

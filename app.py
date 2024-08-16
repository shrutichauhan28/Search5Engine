
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv

# Custom CSS to style the app
st.markdown("""
    <style>
        /* Background for the main content */
        .main {
            background-color: #ADD8E6;
            color: black;
        }
        
        /* Style for the sidebar */
        .css-1d391kg {
            background-color: grey !important;
        }
        
        /* Sidebar titles and text color */
        .css-1d391kg .css-1lcbmhc, .css-1d391kg .css-znku1x {
            color: white !important;
        }
        
        /* Chat message box */
        .stTextArea textarea {
            background-color: #1a1a1a;
            color: white;
        }
        
        /* User messages */
        .st-chat-message-user {
            background-color: #333;
            color: white;
        }
        
        /* Assistant messages */
        .st-chat-message-assistant {
            background-color: #004080;
            color: white;
        }
        
        /* Adjusting title color */
        h1 {
            color:blue;
        }
        
        /* Adjusting link color */
        a {
            color: #00FFFF;
        }
            
             /* Style for the voice search button */
        .voice-button {
            background-color: #00FF00;
            border: none;
            color: white;
            padding: 10px 15px;
            text-align: center;
            font-size: 14px;
            cursor: pointer;
            margin-left: 10px;
        }

        .voice-button:hover {
            background-color: #00cc00;
        }
    </style>
""", unsafe_allow_html=True)




# Arxiv and Wikipedia Tools
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper)

search = DuckDuckGoSearchRun(name="Search")

st.title("🔎 LangChain - Chat with search")
"""
In this example, we're using `StreamlitCallbackHandler` to display the thoughts and actions of an agent in an interactive Streamlit app.
Try more LangChain 🤝 Streamlit Agent examples at [github.com/langchain-ai/streamlit-agent](https://github.com/langchain-ai/streamlit-agent).
"""

# Sidebar for settings
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a chatbot who can search the web. How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

if prompt := st.chat_input(placeholder="What is machine learning?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    llm = ChatGroq(groq_api_key=api_key, model_name="Llama3-8b-8192", streaming=True)
    tools = [search, arxiv, wiki]

    search_agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handling_parsing_errors=True)

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = search_agent.run(st.session_state.messages, callbacks=[st_cb])
        st.session_state.messages.append({'role': 'assistant', "content": response})
        st.write(response)

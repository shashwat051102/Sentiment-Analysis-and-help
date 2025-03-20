from transformers import pipeline
import streamlit as st
from langchain_groq import ChatGroq  # Ensure this import is correct and the module is installed
from langchain_community.utilities import WikipediaAPIWrapper
# DuckDuckGoSearchRun helps you to search anything on the internet
from langchain_community.tools import WikipediaQueryRun,DuckDuckGoSearchRun,BraveSearch
from langchain.agents import initialize_agent,AgentType
# StreamlitCallbackHandler it is used to handle all the tools and agents within the webapp
from langchain.callbacks import StreamlitCallbackHandler
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools import YouTubeSearchTool


import os
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
sentiment_pipeline = pipeline("sentiment-analysis")

input = st.sidebar.text_input("Enter what you are feeling")

result = sentiment_pipeline(input)

output = st.sidebar.write(result[0]['label'])

# Creating custom wrappers
loader = WebBaseLoader("https://www.medicalnewstoday.com/articles/mens-mental-health#common-conditions")
docs = loader.load()
documents = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
documents = documents.split_documents(docs)
embeddings = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")
vectordb = FAISS.from_documents(documents, embedding=embeddings)
retriever = vectordb.as_retriever()
retriever_tool = create_retriever_tool(retriever, "Mental-1", f"Search for information about mental health based on {input} and help the user")

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max = 200)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

# search = BraveSearch.from_api_key(api_key=api_key, search_kwargs={"count": 1})
search = DuckDuckGoSearchRun(name = "Search")
You_tube= YouTubeSearchTool()


if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role":"psychologist","content":"How can I help you?"}
    ]

#
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt:=st.chat_input(placeholder="How can I help you?"):
    st.session_state.messages.append({"role":"user","content":f"Consider {input} and {output} and give answer to the {prompt} given"})
    st.chat_message("user").write(prompt)

    llm = ChatGroq(groq_api_key=api_key, model_name="Llama-3.3-70b-Specdec", streaming=True)
    tools = [search,wiki,You_tube]
    
    search_agents = initialize_agent(tools,llm,agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION,handling_parsing_errors=True)
    
    with st.chat_message("psychologist"):
        st_cb = StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
        response = search_agents.run(st.session_state.messages,callbacks=[st_cb])
        st.session_state.messages.append({"role":"psychologist","content":response})
        st.write(response)
# making non significant changes
# this is only how to use the text embeddings in LLMs
import streamlit as st
import langchain
from langchain.embeddings import OpenAIEmbeddings
import os
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_icon=":robot:", page_title="Similarity Search")
st.header("Hey!! ASK me something and I will give back similar things!!!")

embeddings = OpenAIEmbeddings()

from langchain.document_loaders.csv_loader import CSVLoader
loader = CSVLoader(file_path='myData.csv')
data = loader.load()

db = FAISS.from_documents(data, embeddings)

def get_text():
    input_text = st.text_input("You :", key= input)
    return input_text

user_input = get_text()

submit = st.button("Generate similar things!!")

if submit:
    docs = db.similarity_search(user_input)
    st.subheader("Top matches for you !!!")
    st.text(docs[0])
    st.text(docs[1].page_content)
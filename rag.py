import os
from dotenv import load_dotenv
import faiss
import pickle
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS
# from langchain.retrievers import FaissRetriever

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# loader = DirectoryLoader("./docs", glob="**/*.md", loader_cls=TextLoader)
# docs = loader.load()
#
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# splits = text_splitter.split_documents(docs)
# vectorstore = FAISS.from_documents(documents=splits, embedding=OpenAIEmbeddings())

# faiss.write_index(vectorstore.index, "docs.index")
# vectorstore.save_local("faiss_index")

# vectorstore.index = None
# with open("faiss_store.pkl", "wb") as f:
#     pickle.dump(vectorstore, f)

# index = faiss.read_index('docs.index')
# retriever = FaissRetriever(index=index)

vectorstore = FAISS.load_local(
    "faiss_index", OpenAIEmbeddings(), allow_dangerous_deserialization=True
)
retriever = vectorstore.as_retriever()

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

response = rag_chain.invoke({"input": "How much memory does RTX 4090 have?"})
print(response["answer"])
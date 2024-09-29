import os
import pickle
import faiss
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")

from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

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


loader = DirectoryLoader("./docs", glob="**/*.md", loader_cls=TextLoader)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
# vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
vectorstore = FAISS.from_documents(documents=splits, embedding=OpenAIEmbeddings())

# store = FAISS.from_texts(docs, OpenAIEmbeddings(), metadatas=metadatas)


# faiss.write_index(vectorstore.index, "docs.index")
# vectorstore.index = None
# with open("faiss_store.pkl", "wb") as f:
#     pickle.dump(vectorstore, f)

# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever()

# prompt = hub.pull("rlm/rag-prompt")
#
#
# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)
#
#
# rag_chain = (
#     {"context": retriever | format_docs, "question": RunnablePassthrough()}
#     | prompt
#     | llm
#     | StrOutputParser()
# )

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# print(rag_chain.invoke("How much memory 4090 does have?"))

response = rag_chain.invoke({"input": "How much memory does RTX 4090 have?"})
print(response["answer"])
# for document in response["context"]:
#     print(document)
#     print()
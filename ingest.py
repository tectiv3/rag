# import pickle
import faiss
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")
loader = DirectoryLoader("./docs", glob="**/*.md", loader_cls=TextLoader)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

vectorstore = FAISS.from_documents(documents=splits, embedding=OpenAIEmbeddings())
vectorstore.save_local("faiss_index")

# store = FAISS.from_texts(docs, OpenAIEmbeddings(), metadatas=metadatas)


# faiss.write_index(vectorstore.index, "docs.index")
# vectorstore.index = None
# with open("faiss_store.pkl", "wb") as f:
#     pickle.dump(vectorstore, f)

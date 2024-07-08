import logging
import sys
import openai
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

openai.api_key = ""
import chromadb
from chromadb.db.base import UniqueConstraintError
from llama_index.llms.openai import OpenAI

from llama_index.core import VectorStoreIndex,SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
import chromadb.utils.embedding_functions as embedding_functions
from llama_index.core.memory import ChatMemoryBuffer

openai_ef = embedding_functions.OpenAIEmbeddingFunction(api_key="")
documents = SimpleDirectoryReader(input_files= ["rag_system.txt"]).load_data()
llm = OpenAI()

db = chromadb.PersistentClient(path="C:/Users/AKHIL/PycharmProjects/.cache/chroma")
try:
    chroma_collection = db.create_collection(name='about_rag', embedding_function=openai_ef)
except UniqueConstraintError:
    chroma_collection = db.get_collection(name='about_rag', embedding_function=openai_ef)
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)



memory = ChatMemoryBuffer.from_defaults(token_limit=300)
chat_engine = index.as_chat_engine(chat_mode="react",memory=memory, llm=llm, verbose=True)
response = chat_engine.chat("What is rag fullform")
response=chat_engine.chat("list top 5 advantages  of rag")
response = chat_engine.chat("What did I ask you before?")
print(response)


chat_engine.reset()




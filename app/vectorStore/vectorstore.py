from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

OPENAI_API_KEY = api_key

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set")

embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

def get_vectorstore(conversation_id: str):
    try:
        from langchain_chroma import Chroma
        return Chroma(
            collection_name=conversation_id,
            embedding_function=embeddings,
            persist_directory=f"./chroma_storage/{conversation_id}",
        )
    except Exception as e:
        from langchain_community.vectorstores import Chroma
        return Chroma(
            collection_name=conversation_id,
            embedding_function=embeddings,
            persist_directory=f"./chroma_storage/{conversation_id}",
        )
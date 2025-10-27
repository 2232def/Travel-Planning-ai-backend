from langgraph.graph import MessagesState
# from langchain.chat_models import init_chat_model
from .retriever import QdrantRetriever
from dotenv import load_dotenv
from langchain_core.tools import tool
import os

from langchain_google_genai import ChatGoogleGenerativeAI

import getpass
import os



def init_chat_model(model_name: str  = "gemini-2.5-flash", temperature: float = 0) -> ChatGoogleGenerativeAI:
    if "GOOGLE_API_KEY" not in os.environ:
        os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")
        
    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
    )

load_dotenv()

_retriever = QdrantRetriever(
        collection_name=os.getenv("QDRANT_COLLECTION", "travel_plans"),
        top_k=5,
)

@tool
def qdrant_retrieve(query: str, k: int = 5) -> list[dict]:
    """Retrieve relevant documents from the Qdrant vector database."""
    results = _retriever.search(query, k=k)
    return results


# response_model = init_chat_model("gemini-2.5-flash", temperature=0)
model_with_tools = init_chat_model().bind_tools([qdrant_retrieve])

def generate_query_or_respond(state: MessagesState):
    """Call the model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply respond to the user.
    """
    # response = (
    #     response_model
    #     .bind_tools([]).invoke(state["messages"])
    # )
    response = model_with_tools.invoke(state["messages"])
    return {"messages": [response]}
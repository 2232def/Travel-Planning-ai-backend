import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
# from sympy import limit
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, QueryRequest, NamedVector
from langchain_ollama import OllamaEmbeddings
import numpy

load_dotenv()

# VECTOR_NAME = os.getenv("QDRANT_VECTOR_NAME", "text_embedding")

class QdrantRetriever:
    def __init__(
        self,
        collection_name: str,
        top_k: int = 5,
        embed_model: str = "nomic-embed-text",
        ollama_base_url: Optional[str] = None,
    ):
        self.collection = collection_name
        self.top_k = top_k
        self.client = QdrantClient(
            url=os.getenv("QDRANT_URL", "https://7e872273-4048-4107-832b-f154bdda1cf3.europe-west3-0.gcp.cloud.qdrant.io:6333"),
            api_key=os.getenv("QDRANT_API_KEY"),
        )
        kwargs = {}
        if ollama_base_url or os.getenv("OLLAMA_BASE_URL"):
            kwargs["base_url"] = ollama_base_url or os.getenv("OLLAMA_BASE_URL")
        self.embedder = OllamaEmbeddings(model=embed_model, **kwargs)

    def embed_query(self, query: str) -> List[float]:
        return self.embedder.embed_query(query)

    def search(self, query: str, k: Optional[int] = None, qdrant_filter: Optional[Filter] = None) -> List[Dict[str, Any]]:
        vec = self.embed_query(query)
        if isinstance(vec, (numpy.ndarray,)):
            vec = vec.tolist()
        limit = k or self.top_k
        # query_vector = numpy.random.rand(100)
        # Use the new query_points API


        response = self.client.query_points(
            collection_name=self.collection,
            query=vec,
            limit=limit,
            # filter=qdrant_filter,

            # limit=k or self.top_k,
            with_payload=True,
    )

        return [
            {
                "id": str(point.id),
                "score": float(point.score) if hasattr(point, 'score') else 0.0,
                "text": (point.payload or {"text": "No result"}).get("text", ""),
                "metadata": point.payload or {},
                # "metadata": point.payload or {},
            }
            for point in response.points
        ]


if __name__ == "__main__":
    _retriever = QdrantRetriever(
        collection_name=os.getenv("QDRANT_COLLECTION", "travel_plans"),
    )
    query = "Where is Jamshedpur located?"
    results = _retriever.search(query, k=2)
    print("Query:", query)
    print("Results:", results)
    print(f"Found {len(results)} results")
    for i, r in enumerate(results, 1):
        print(f"{i}. score={r['score']:.4f}")
        print(r["text"][:200], "...\n")
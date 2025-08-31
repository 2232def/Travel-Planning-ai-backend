from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
import os
load_dotenv()
from app.services.embeddings import TravelEmbeddingPipeline


qdrant_client = QdrantClient(
    url="https://7e872273-4048-4107-832b-f154bdda1cf3.europe-west3-0.gcp.cloud.qdrant.io:6333", 
    api_key=os.getenv("QDRANT_API_KEY"),
)

# pipeline = TravelEmbeddingPipeline()

def create_collection(collection_name):
    qdrant_client.create_collection(
        collection_name,
        vectors_config={
            "text_embedding": models.VectorParams(
                size=768,
                distance=models.Distance.COSINE
            )
        }
    )




def ingest_data(points): 
    operation_info = qdrant_client.upsert(
        collection_name="travel_plans",
        points=points
    )
    return operation_info



if __name__ == "__main__":
    create_collection("travel_plans")
    out = TravelEmbeddingPipeline(use_embeddings=True).run()
    points = []
    embeddings = out.get("embeddings")
    if embeddings is not None:
        for i, embedding in enumerate(embeddings):
            points.append(models.PointStruct(id=i, vector={"text_embedding": embedding}))
        print(f"Prepared {len(points)} points for ingestion.")
    else:
        raise ValueError("Embeddings data is missing or None.")
    ingest_data(points)
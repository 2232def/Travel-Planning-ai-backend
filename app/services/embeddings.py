# This file contains the code for generating embeddings.
from langchain_text_splitters import RecursiveCharacterTextSplitter
from bs4 import Tag
from langchain_text_splitters import HTMLSemanticPreservingSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import PromptTemplate
from ..utils.web_loader import WebLoader

# WebLoader().load()
# print("WebLoader initialized and documents loaded successfully." , WebLoader().load())
class TravelEmbeddingPipeline:
    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 120,
        # embed_model: str = "models/text-embedding-004",  # Google GenAI embeddings
        # use_embeddings: bool = False,
    ):
        self.loader = WebLoader()
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
            separators=[
                "\n\n",
                "\n",
                " ",
                ".",
                ",",
                "\u200b",
                "\uff0c",
                "\u3001",
                "\uff0e",
                "\u3002",
            ],  # no empty string to avoid regex issues
        )
        # self.embed_model = embed_model
        # self.use_embeddings = use_embeddings
        # self._embedder: Optional[GoogleGenerativeAIEmbeddings] = None

    # def _ensure_embedder(self):
    #     if not self.use_embeddings:
    #         return
    #     if not os.getenv("GOOGLE_API_KEY"):
    #         raise RuntimeError("GOOGLE_API_KEY not set. Add it to your .env.")
    #     if self._embedder is None:
    #         self._embedder = GoogleGenerativeAIEmbeddings(model=self.embed_model)

    def load_docs(self):
        # Returns List[Document]
        return self.loader.load()

    def split_docs(self, docs):
        # Returns List[Document] chunks
        return self.splitter.split_documents(docs)

    def to_texts(self, docs_or_chunks) -> List[str]:
        return [d.page_content for d in docs_or_chunks]

    # def embed_texts(self, texts: List[str]):
    #     self._ensure_embedder()
    #     if not self._embedder:
    #         return []
    #     return self._embedder.embed_documents(texts)

    def run(self):
        docs = self.load_docs()
        chunks = self.split_docs(docs)
        texts = self.to_texts(chunks)
        embeddings = self.embed_texts(texts) if self.use_embeddings else None
        return {
            "docs": docs,
            "chunks": chunks,
            "texts": texts,
            "embeddings": embeddings,
        }


# if __name__ == "__main__":
#     pipeline = TravelEmbeddingPipeline(use_embeddings=False)  # set True if GOOGLE_API_KEY is configured
#     out = pipeline.run()
#     print(f"docs={len(out['docs'])}, chunks={len(out['chunks'])}")
#     if out["texts"]:
#         print(f"first chunk chars={len(out['texts'][0])}, words={len(out['texts'][0].split())}")
#     if out["embeddings"] is not None:
#         print(f"embeddings computed: {len(out['embeddings'])}")

# headers_to_split_on = [
#     ("h1", "Header 1"),
#     ("h2", "Header 2"),
# ]

# def code_handler(element: Tag) -> str:
#     data_lang = element.get("data-lang")
#     code_format = f"<code:{data_lang}>{element.get_text()}</code>"

#     return code_format


# splitter = HTMLSemanticPreservingSplitter(
#     headers_to_split_on=headers_to_split_on,
#     separators=["\n\n", "\n", ". ", "! ", "? "],
#     max_chunk_size=50,
#     preserve_images=True,
#     preserve_videos=True,
#     elements_to_preserve=["table", "ul", "ol", "code"],
#     denylist_tags=["script", "style", "head"],
#     custom_handlers={"code": code_handler},
# )

# documents = splitter.split_text()
# documents 
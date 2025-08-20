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
text = WebLoader().load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
    separators=[
        "\n\n",
        "\n",
        " ",
        ".",
        ",",
        "\u200b",  # Zero-width space
        "\uff0c",  # Fullwidth comma
        "\u3001",  # Ideographic comma
        "\uff0e",  # Fullwidth full stop
        "\u3002",  # Ideographic full stop
        "",
    ],
)
texts = text_splitter.split_documents(text)
print(texts[0])
print(texts[1])
print(texts[2])
print(texts[3])

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
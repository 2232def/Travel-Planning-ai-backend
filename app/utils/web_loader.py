import os
from langchain_community.document_loaders import WebBaseLoader
from urls import urls

USER_AGENT = os.getenv("USER_AGENT", "TravelPlanningAI/1.0 (+https://example.com/contact)")

class WebLoader:
    def __init__(self):
        self.loader = WebBaseLoader(urls(),header_template={"User-Agent" : USER_AGENT})

    def load(self):
        return self.loader.load()

if __name__ == "__main__":
    docs = WebLoader().load()
    print(docs[1])
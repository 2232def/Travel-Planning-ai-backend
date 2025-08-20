import os
from langchain_community.document_loaders import WebBaseLoader
from .urls import urls

USER_AGENT = os.getenv("USER_AGENT", "TravelPlanningAI/1.0 (+https://example.com/contact)")
os.environ["USER_AGENT"] = USER_AGENT

class WebLoader:
    def __init__(self):
        self.loader = WebBaseLoader(urls(),header_template={"User-Agent" : USER_AGENT})

    def load(self):
        return self.loader.load()

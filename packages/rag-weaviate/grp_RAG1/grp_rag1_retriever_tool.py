from langchain.tools.retriever import create_retriever_tool
from langchain_community.vectorstores import Weaviate
import weaviate


import os


client = weaviate.Client(
  url=os.getenv('WEAVIATE_CLIENT_URL'),
)

vectorstore = Weaviate(client, 
                       os.getenv('WEAVIATE_CLASS_NAME'), 
                       os.getenv('WEAVIATE_CLASS_PROPERTY'))

retriever = vectorstore.as_retriever()



retriever_tool = create_retriever_tool(
    retriever,
    "mediwave_search",
    "Search any information only about mediwave or mindwave. For any questions related to Mediwave or mindwave, you must use this tool!",
)
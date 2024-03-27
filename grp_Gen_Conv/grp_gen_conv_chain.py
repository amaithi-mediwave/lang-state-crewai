
def general_conversation(state):
  
  input = state["messages"][0].content
  # print(input)


  import os
  from dotenv import load_dotenv
  load_dotenv()

  from langchain_community.vectorstores import Weaviate
  from langchain_core.output_parsers import StrOutputParser
  from langchain_core.prompts import ChatPromptTemplate
  from langchain_core.pydantic_v1 import BaseModel
  from langchain_core.runnables import RunnableParallel, RunnablePassthrough
  from langchain_community.chat_models import ChatOllama

  import weaviate
  from langchain.globals import set_llm_cache
  from langchain.cache import RedisCache
  import redis


  redis_client = redis.Redis.from_url(os.environ['REDIS_URL'])
  set_llm_cache(RedisCache(redis_client))

  # Chat prompt
  template = """You're an Friendly AI assistant, your name is Claro, you can make normal conversations in a friendly manner, and also provide Answer the question and make it sounds like human

  Question: {question}
  """
  prompt = ChatPromptTemplate.from_template(template)




  # RAG
  model = ChatOllama(model=os.environ['LLM'])

  chain = (
    
      {"question": RunnablePassthrough()}
      | prompt
      | model
      # | StrOutputParser()
  )



  # Add typing for input
  class Question(BaseModel):
      __root__: str


  chain = chain.with_types(input_type=Question)
  
  result = chain.invoke(input)
  # print({"messages": [result]})
  
  return {"messages": [result], 'next': 'supervisor'}
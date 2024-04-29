

def mediwave_rag(state):
    input = state["messages"][0].content
    
    import os
    import warnings
    warnings.filterwarnings('ignore')
    from langchain_community.vectorstores import Weaviate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.pydantic_v1 import BaseModel
    from langchain_core.runnables import RunnableParallel, RunnablePassthrough
    from langchain_core.agents import AgentFinish
    from langchain_groq.chat_models import ChatGroq


    import weaviate
    from langchain.globals import set_llm_cache
    from langchain.cache import RedisCache
    import redis


    REDIS_URL = os.environ['REDIS_URL']
    WEAVIATE_CLIENT_URL = os.environ['WEAVIATE_CLIENT_URL']
    WEAVIATE_CLASS_NAME = os.environ['WEAVIATE_CLASS_NAME']
    WEAVIATE_CLASS_PROPERTY = os.environ['WEAVIATE_CLASS_PROPERTY']
    
    
    redis_client = redis.Redis.from_url(REDIS_URL)
    
    set_llm_cache(RedisCache(redis_client))


    weaviate_client = weaviate.Client(
    url= WEAVIATE_CLIENT_URL
    )

    vectorstore = Weaviate(weaviate_client, 
                        WEAVIATE_CLASS_NAME, 
                        WEAVIATE_CLASS_PROPERTY)

    retriever = vectorstore.as_retriever()


    # RAG prompt
    template = """You're an Friendly AI assistant, your name is Claro, you can make normal conversations in a friendly manner, and also provide Answer the question based on the following context make sure it sounds like human and official assistant:
    {context}
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)


    # RAG
    model = ChatGroq(
                model=os.environ['LLM'],
                api_key=os.environ['GROQ_API_KEY']
                )

    chain = (
        RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
        | prompt
        | model
        | StrOutputParser()
    )


    # Add typing for input
    class Question(BaseModel):
        __root__: str


    chain = chain.with_types(input_type=Question)

    result = chain.invoke(input)
        
    return {"agent_outcome": AgentFinish(return_values={'output': result}, log=result)}

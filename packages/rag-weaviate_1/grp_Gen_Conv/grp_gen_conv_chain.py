







def general_conversation(state):
  
    input = state["messages"][0].content

    import os
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.pydantic_v1 import BaseModel
    from langchain_core.runnables import RunnablePassthrough
    from langchain.globals import set_llm_cache
    from langchain.cache import RedisCache
    import redis
    from langchain_core.agents import AgentFinish

    from langchain_groq.chat_models import ChatGroq

    redis_client = redis.Redis.from_url(os.environ['REDIS_URL'])
    
    set_llm_cache(RedisCache(redis_client))

    # Chat prompt
    template = """You're an Friendly AI assistant, your name is Claro, you can make normal conversations in a friendly manner, and also provide Answer the question and make it sounds like human

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)


    # RAG
    # model = ChatOllama(model=os.environ['LLM'])
    model = ChatGroq(
        model=os.environ['LLM'],
        api_key=os.environ['GROQ_API_KEY']
        )

    chain = (

        {"question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )


    # Add typing for input
    class Question(BaseModel):
        __root__: str

    chain = chain.with_types(input_type=Question)

    result = chain.invoke(input)

    # return {"messages": [result], 'next': 'supervisor'}

    return {"agent_outcome": AgentFinish(return_values={'output': result}, log=result)}


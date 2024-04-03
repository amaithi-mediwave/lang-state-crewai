
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_experimental.llms.ollama_functions import OllamaFunctions
import os 

# members = ["Food_crew", "General_conversation", "General_other", "Mediwave_rag", "Travel_crew"]



members = ["Food_crew", "General_conv", "General_other", "Mediwave_rag", "Travel_crew"]

def supervisor_node(state):

    from langchain.globals import set_debug, set_verbose

    set_debug=True 
    set_verbose=True


    system_prompt = (
        """You are a supervisor tasked with managing a conversation between the
        following workers:  {members}. Given the following user request,"
        respond with the worker to act next. 
        
        if the user asks anything related to food, receipies, and it's related stuffs use 'Food_crew',
        if the user asks anything related to mediwave and it's related stuffs use 'Mediwave_rag',    
        if the user makes conversation, jokes and funny conversations then use 'General_conv',
        if the user asks anything related to weather, time, wikipedia and it's related stuffs use 'General_other',
        if the user asks anything related to travel, exploration, city tour and it's related stuffs use 'Travel_crew'
            
        Each worker will perform a
        task and respond with their results and status. When finished,
        respond with FINISH."""
    )

    # Our team supervisor is an LLM node. It just picks the next agent to process
    # and decides when the work is completed
    options = ["FINISH"] + members
    # Using openai function calling can make output parsing easier for us
    function_def = {
        "name": "route",
        "description": "Select the next role to act",
        
        "parameters": {
            "type": "object",        
            "properties": {
                "next": {
                    "type": "string",
                    "enum": f"{options}",
                }
            },
            "required": ["next"],
        },
    }


    DEFAULT_SYSTEM_TEMPLATE = """You have access to the following tools:

    {tools}

    You must always select one of the above tools and respond with only a JSON object matching the following schema:

    {{
    "tool": "route",
    "tool_input": <parameters for the selected tool, matching the tool's JSON schema>
    }}
    """ 


    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            (
                "system",
                "Given the conversation above, who should act next?"
                " Or should we FINISH? Select one of: {options}",
            ),
        ]
    ).partial(options=str(options), members=", ".join(members))

    llm = OllamaFunctions(
        model= os.environ['LLM'],
        tool_system_prompt_template=DEFAULT_SYSTEM_TEMPLATE
        )

    # print(state)
    
    supervisor_chain = (
        # {"messages": state['messages'][0]}|
        prompt
        | llm.bind(functions=[function_def], function_call={"name": "route"})
        | JsonOutputFunctionsParser()
    ).with_config({"run_name": "Supervisor"})
    
    result = supervisor_chain.invoke(state)
    # print(result)
    
    return result

# from langchain_core.pydantic_v1 import BaseModel
# import os
# from langchain_core.pydantic_v1 import BaseModel, Field
# from typing import List
# from langchain_experimental.llms.ollama_functions import OllamaFunctions

# class Plan(BaseModel):
#     """Plan to follow in future"""

#     steps: List[str] = Field(
#         description="different steps to follow, should be in sorted order"
#     )
    
    
# from langchain.chains.openai_functions import create_structured_output_runnable
# from langchain_core.prompts import ChatPromptTemplate

# planner_prompt = ChatPromptTemplate.from_template(
#     """For the given objective, come up with a simple step by step plan but don't provide answer coz you have tools to figure out things. \
# This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
# The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

# if the given objective related to mediwave then give the objective as plan

# {objective}"""
# )
# planner = create_structured_output_runnable(
#     Plan, 
#     OllamaFunctions(model=os.environ['LLM']),
#     planner_prompt
# )


# ----------------------------------- V2 ---------------------------------------

from langchain_core.pydantic_v1 import BaseModel, Field
from typing_extensions import List
from typing import Literal

class Step(BaseModel):
    key: Literal["Food_crew", "General_conv", "General_other", "Mediwave_rag", "Travel_crew"] = Field(description='the worker gonna handle this task/step')
    value: str = Field(description='task/ step the worker need to do')


class Plan(BaseModel):
    """Plan to follow in future"""

    steps: List[Step] 
    


crews = ["Food_crew", "General_conv", "General_other", "Mediwave_rag", "Travel_crew"]

function = {'name': 'plan',
 'description': '',
 
 'parameters': {
   'type': 'array',
   'properties': {
     'key': {
       "enum": f"{crews}",
       'description': 'the worker gonna handle this task/step',
       'type': 'string'},
     'value': {
       'description': 'task/ step the worker need to do',
      'type': 'string'}
     },
                    
    'required': ['plan'],
  }
}


from langchain_core.pydantic_v1 import BaseModel
import os
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain_community.chat_models.ollama import ChatOllama
    
from langchain.chains.openai_functions import create_structured_output_runnable
from langchain_core.prompts import ChatPromptTemplate

planner_prompt = ChatPromptTemplate.from_template(
    """For the given user input, come up with a simple step by step plan but don't provide answer coz you have tools to figure out things. if the input requires multiple steps(combintion of multiple tools) then create a list or else just give direct single step with proper input to the selected tool.\
        
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
    
    if the user asks anything related to food, receipies, and it's related stuffs use 'Food_crew' key,
        if the user asks anything related to mediwave and it's related stuffs use 'Mediwave_rag' key,    
        if the user makes conversation like hi, hello and something like that, jokes and funny conversations then use 'General_conv' key,
        if the user asks anything related to weather, time, wikipedia and it's related stuffs use 'General_other' key,
        if the user asks anything related to travel, exploration, city tour and it's related stuffs use 'Travel_crew' key.
    
    
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

make sure the tool name is always 'output_formatter'


user input : {objective}"""
)

llm = ChatOllama(model=os.environ['LLM'], stop= [
        os.environ['LLM_START_PARAM'],
        os.environ['LLM_STOP_PARAM']
    ]
                 )


planner = create_structured_output_runnable(
    function, 
    OllamaFunctions(llm=llm),
    planner_prompt,
    enforce_function_usage = True,
    )
# ).with_retry(
#   retry_if_exception_type = (ValueError, KeyError),
#   stop_after_attempt = 4
# )


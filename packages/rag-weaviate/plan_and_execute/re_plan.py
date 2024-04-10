# from langchain.chains.openai_functions import create_openai_fn_runnable
# from langchain_core.pydantic_v1 import BaseModel
# from langchain_core.prompts import ChatPromptTemplate
# from plan_and_execute.planner import crews
# from langchain_experimental.llms.ollama_functions import OllamaFunctions
# import os


# from langchain_core.pydantic_v1 import BaseModel, Field
# from typing_extensions import List



# class Step(BaseModel):
#     key: str = Field(description='the worker gonna handle this task/step')
#     value: str = Field(description='task/ step the worker need to do')


# class Plan(BaseModel):
#     """Plan to follow in future"""

#     steps: List[Step] 
    






# class Response(BaseModel):
#     """Response to user."""

#     response: str

# response = {'name': 'Response',
#  'description': 'Response to user.',
#  'parameters': {'type': 'object',
#   'properties': {'response': {'type': 'string'}},
#   'required': ['response']}}


# function1 = {'name': 'plan',
#  'description': 'replanner',
 
#  'parameters': {
#    'type': 'array',
#    'properties': {
#      'key': {
#        "enum": f"{crews}",
#        'description': 'the worker gonna handle this task/step',
#        'type': 'string'},
#      'value': {
#        'description': 'task/ step the worker need to do',
#       'type': 'string'}
#      },
                    
#     'required': ['plan'],
#   }
# }



# replanner_prompt = ChatPromptTemplate.from_template(
#     """For the given objective, come up with a simple step by step plan. \
# This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
    
    
#      if the objective is anything related to food, receipies, and it's related stuffs use 'Food_crew' worker,
#        if the objective is anything related to mediwave and it's related stuffs use 'Mediwave_rag' worker,    
#         if the objective makes conversation, jokes and funny conversations then use 'General_conv' worker,
#         if the objective is anything related to weather, time, wikipedia and it's related stuffs use 'General_other' worker,
#         if the objective is anything related to travel, exploration, city tour and it's related stuffs use 'Travel_crew' worker.
    
# The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

# Your objective was this:
# {input}

# Your original plan was this:
# {plan}

# You have currently done the follow steps:
# {past_steps}

# Update your plan accordingly(remove the completed step). If no more steps are needed and you can return to the user, then respond with that. Otherwise, fill out the plan. Only add steps to the plan that still NEED to be done. Do not return previously done steps as part of the plan."""
# )


# replanner = create_openai_fn_runnable(
    
#     [Plan, Response],
#     OllamaFunctions(model=os.environ['LLM']),
#     replanner_prompt,
# )

# # [function1, response],


# ------------------------------------------ v2 --------------------------






from langchain_core.pydantic_v1 import BaseModel, Field
from typing_extensions import List
from typing import Literal

class Step(BaseModel):
    key: Literal["Food_crew", "General_conv", "General_other", "Mediwave_rag", "Travel_crew"] = Field(description='the worker gonna handle this task/step')
    value: str = Field(description='task/ step the worker need to do')


class Plan(BaseModel):
    """Plan to follow in future"""

    steps: List[Step] 
    





from langchain.chains.openai_functions import create_openai_fn_runnable
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from plan_and_execute.planner import crews
from langchain_experimental.llms.ollama_functions import OllamaFunctions
import os
from langchain_community.chat_models.ollama import ChatOllama

from langchain.globals import set_debug, set_verbose


set_verbose=True 
set_debug=True

class Response(BaseModel):
    """Response to user."""

    response: str

response = {'name': 'Response',
 'description': 'Response to user.',
 'parameters': {'type': 'object',
  'properties': {'response': {'type': 'string'}},
  'required': ['response']}}


function1 = {'name': 'plan',
 'description': 'replanner',
 
 'parameters': {
   'type': 'array',
   'properties': {
     'key': {
       "enum": f"{crews}",
       'description': 'the worker gonna handle this task/step'
       
,
        
       'type': 'string'},
     'value': {
       'description': 'task/ step the worker need to do',
      'type': 'string'}
     },
                    
    'required': ['plan'],
  }
}



replanner_prompt = ChatPromptTemplate.from_template(
    """For the given user input, come up with a simple step by step plan. \
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
    
      if the user asks anything related to food, receipies, and it's related stuffs use 'Food_crew' key,
        if the user asks anything related to mediwave and it's related stuffs use 'Mediwave_rag' key,    
        if the user makes conversation, jokes and funny conversations then use 'General_conv' key,
        if the user asks anything related to weather, time, wikipedia and it's related stuffs use 'General_other' key,
        if the user asks anything related to travel, exploration, city tour and it's related stuffs use 'Travel_crew' key.
    
    
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

user input was this:
{input}

Your original plan was this:
{plan}

You have currently done the follow steps:
{past_steps}

Update your plan accordingly(remove the completed step). If no more steps are needed and you can return to the user, then respond with that. Otherwise, fill out the plan. Only add steps to the plan that still NEED to be done. Do not return previously done steps as part of the plan.

only provide the final answer after make sure the user requirement has been satisfied completely.

make sure the tool name is either 'Plan' or 'Response'

while providing response make sure the user input is satisfied with the response refer the follow steps to gather the necessary informations for the final response.
"""
)

llm = ChatOllama(model=os.environ['LLM'], stop= [
       "[INST]",
        "[/INST]"
    ]
                 )


replanner = create_openai_fn_runnable(
    
    [Plan, Response],
    OllamaFunctions(llm=llm),
    replanner_prompt,
).with_retry(
  retry_if_exception_type=(ValueError,KeyError),
  stop_after_attempt=4,
)



# output = replanner.invoke(state_)


# if isinstance(output, Response):
#     print({"response": output})
# else:
#     print({"plan": output})

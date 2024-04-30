
from langchain_core.pydantic_v1 import BaseModel, Field
from typing_extensions import List
from typing import Literal
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.prompts import ChatPromptTemplate
import os

from langchain.globals import set_debug, set_verbose
from langchain_groq.chat_models import ChatGroq
from langchain_core.output_parsers.openai_tools import PydanticToolsParser


set_verbose=True 
set_debug=True

class Step(BaseModel):
    key: Literal["Food_crew", "General_conv", "General_other", "Mediwave_rag", "Travel_crew"] = Field(description='the worker gonna handle this task/step')
    value: str = Field(description='task/ step the worker need to do')


class Plan(BaseModel):
    """Plan to follow in future"""

    steps: List[Step] 
    
class Response(BaseModel):
    """Response to user."""

    response: str


replanner_prompt = ChatPromptTemplate.from_template(
   """
you are a planning expert following are the user input, and the plan as well as the performed steps and it's outcomes you're job is to analyse the currently executed steps and its outcome to summarize the final answer to the user question, while providing response make sure the user input is satisfied with the response refer the follow steps to gather the necessary informations for the final response..

user input was this:
{input}

Your original plan was this:
{plan}

You have currently done the follow steps:
{past_steps}

\n incase you need more information/ execute all steps in the plan to summarize the final answer then do the following.

Update your plan accordingly(remove the completed step). If no more steps are needed and you can return to the user, then respond with that. Otherwise, fill out the plan. Only add steps to the plan that still NEED to be done and don't create unnecessary new steps that are not in the original plan. Do not return previously done steps as part of the plan.

only provide the final answer after make sure the user requirement has been satisfied completely.

only use one of the given tool and make sure to follow it's schema. 

make sure the tool name is either 'Plan' for replan or 'Response' for final answer.

don't make unneccesary steps unrelavant to the original user input.. remove the satisfied step in the given plan. 

don't add any notes to the output.
"""
)


llm = ChatGroq(
  model=os.environ['LLM'],
  api_key=os.environ['GROQ_API_KEY']
)

parser = PydanticToolsParser(tools=[Plan, Response], first_tool_only=True)

llm_func = llm.bind_tools([Plan, Response], tool_choice='auto').with_retry(
                                                                        retry_if_exception_type = (ValueError, KeyError),
                                                                        stop_after_attempt = 4
                                                                        )

replanner = replanner_prompt | llm_func | parser



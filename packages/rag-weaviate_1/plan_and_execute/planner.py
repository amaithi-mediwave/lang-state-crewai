
from langchain_core.pydantic_v1 import BaseModel, Field
from typing_extensions import List
from typing import Literal
from dotenv import load_dotenv
load_dotenv()

from langchain_core.pydantic_v1 import BaseModel
import os
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List
from langchain_groq.chat_models import ChatGroq
from langchain_core.prompts import ChatPromptTemplate



class Step(BaseModel):
    key: Literal["Food_crew", "General_conv", "General_other", "Mediwave_rag", "Travel_crew"] = Field(description='the worker gonna handle this task/step')
    value: str = Field(description='task/ step the worker need to do')


class Plan(BaseModel):
    """Plan to follow in future"""

    steps: List[Step] 
    

planner_prompt = ChatPromptTemplate.from_template(
    """For the given user input, come up with a simple step by step plan but don't provide answer coz you have tools to figure out things. if the input requires multiple steps(combintion of multiple tools) then create a list or else just give direct single step with proper input to the selected tool.\
        
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
    
    if the user asks anything related to food, receipies, and it's related stuffs use 'Food_crew' key,
        if the user asks anything related to mediwave and it's related stuffs use 'Mediwave_rag' key,    
        if the user makes conversation like hi, hello and something like that, jokes and funny conversations then use 'General_conv' key,
        if the user asks anything related to weather, time, wikipedia and it's related stuffs use 'General_other' key,
        if the user asks anything related to travel, exploration, city tour and it's related stuffs use 'Travel_crew' key.
    
    always plan minimal steps. make multiple steps only if it is necessary. don't make unnecessary steps, make sure the steps satisfy the original user input.
    
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

make sure the tool name is always 'output_formatter'

only add the actionable step not the empty steps..

don't use ```json``` while giving the response

user input : {objective}"""
)



llm = ChatGroq(
              model=os.environ['LLM'],
              api_key=os.environ['GROQ_API_KEY']
                )

planner = planner_prompt | llm.with_structured_output(
                                                      schema=Plan,
                                                      method='function_calling'
                                                    ).with_retry(
                                                                        retry_if_exception_type = (ValueError, KeyError),
                                                                        stop_after_attempt = 4
                                                                        )

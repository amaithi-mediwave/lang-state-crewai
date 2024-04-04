from langchain_core.pydantic_v1 import BaseModel
import os
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List
from langchain_experimental.llms.ollama_functions import OllamaFunctions

class Plan(BaseModel):
    """Plan to follow in future"""

    steps: List[str] = Field(
        description="different steps to follow, should be in sorted order"
    )
    
    
from langchain.chains.openai_functions import create_structured_output_runnable
from langchain_core.prompts import ChatPromptTemplate

planner_prompt = ChatPromptTemplate.from_template(
    """For the given objective, come up with a simple step by step plan but don't provide answer coz you have tools to figure out things. \
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

if the given objective related to mediwave then give the objective as plan

{objective}"""
)
planner = create_structured_output_runnable(
    Plan, 
    OllamaFunctions(model=os.environ['LLM']),
    planner_prompt
)

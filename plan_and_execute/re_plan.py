from langchain.chains.openai_functions import create_openai_fn_runnable
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from plan_and_execute.planner import Plan
from langchain_experimental.llms.ollama_functions import OllamaFunctions
import os

class Response(BaseModel):
    """Response to user."""

    response: str


replanner_prompt = ChatPromptTemplate.from_template(
    """For the given objective, come up with a simple step by step plan. \
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

Your objective was this:
{input}

Your original plan was this:
{plan}

You have currently done the follow steps:
{past_steps}

Update your plan accordingly. If no more steps are needed and you can return to the user, then respond with that. Otherwise, fill out the plan. Only add steps to the plan that still NEED to be done. Do not return previously done steps as part of the plan."""
)


replanner = create_openai_fn_runnable(
    [Plan, Response],
    OllamaFunctions(model=os.environ['LLM']),
    replanner_prompt,
).with_config({"run_name": "Super Graph - Re Planner"})
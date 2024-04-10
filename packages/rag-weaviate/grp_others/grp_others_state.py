import operator
from typing import Annotated, TypedDict, Union
from langchain import hub
from langchain.agents import create_react_agent
from langchain_community.chat_models import ChatOllama
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage

from .grp_others_tools import tools 
import os


class AgentState(TypedDict):
    input: str
    chat_history: list[BaseMessage]
    agent_outcome: Union[AgentAction, AgentFinish, None]
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]


model = ChatOllama(model=os.getenv('LLM'))

prompt = hub.pull("hwchase17/react")


# prompt = PromptTemplate(
#     input_variables=['agent_scratchpad', 'input', 'tool_names', 'tools'], template="""Answer the following questions as best you can. You have access to the following tools:\n\n{tools}\n
#     \nUse the following format:\n\n
    
#     Question: the input question you must answer\n
    
#     Thought: you should always think about what to do\n
#     Action: the action to take, should be one of [{tool_names}]\n
#     Action Input: the input to the action\n
#     Observation: the result of the action\n... (this Thought/Action/Action Input/Observation can repeat N times)\n
#     Thought: I now know the final answer\n
#     Final Answer: the final answer to the original input question (make sure to use the same case) \n\n
    
#     Begin!\n\n
#     Question: {input}\n
#     Thought:{agent_scratchpad}""")


agent_runnable = create_react_agent(model, tools, prompt).with_retry(
    retry_if_exception_type=(ValueError, KeyError),
    stop_after_attempt=4
)
import operator
from typing import Annotated, TypedDict, Union

from langchain import hub
from langchain.agents import create_react_agent
from langchain_community.chat_models import ChatOllama
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage


from .grp_others_tools import tools 

from dotenv import load_dotenv
load_dotenv()
import os



class AgentState(TypedDict):
    input: str
    chat_history: list[BaseMessage]
    agent_outcome: Union[AgentAction, AgentFinish, None]
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]


model = ChatOllama(model=os.getenv('LLM'))

prompt = hub.pull("hwchase17/react")


agent_runnable = create_react_agent(model, tools, prompt)
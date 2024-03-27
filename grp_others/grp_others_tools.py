import operator
from datetime import datetime
from typing import Annotated, TypedDict, Union

from dotenv import load_dotenv
from langchain import hub
from langchain.agents import create_react_agent
from langchain_community.chat_models import ChatOllama
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolExecutor, ToolInvocation

from langchain_community.utilities import OpenWeatherMapAPIWrapper

from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool
import os

load_dotenv()

# ------------------- CURRENT TIME TOOL -------------------
@tool
def get_now(format: str = "%Y-%m-%d %H:%M:%S"):
    """
    Get the current time
    """
    format="%Y-%m-%d %H:%M:%S"
    return datetime.now().strftime(format)



# ------------------- WEATHER TOOL -------------------
def get_weather(location):
    """
    Get the current weather for a given location"""
    weather = OpenWeatherMapAPIWrapper()
    weather_data = weather.run(location)
    return weather_data


weather = StructuredTool.from_function(
    func=get_weather,
    name='Weather',
    description="useful for getting the current weather data about a partidular given location",
    handle_tool_error=True
)

# -------------------  WIKIPEDIA TOOL --------------

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper


from langchain_core.pydantic_v1 import BaseModel, Field


api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)

class WikiInputs(BaseModel):
    """Inputs to the wikipedia tool."""

    query: str = Field(
        description="query to look up in Wikipedia, should be 3 or less words"
    )
    

wiki = WikipediaQueryRun(
    name="wikipedia_search",
    description="wikipedia_search - to search things in wikipedia",
    args_schema=WikiInputs,
    api_wrapper=api_wrapper,
    # return_direct=True,
)


# -------------------  SEARCH TOOL --------------




tools = [get_now, weather, wiki]

tool_executor = ToolExecutor(tools)
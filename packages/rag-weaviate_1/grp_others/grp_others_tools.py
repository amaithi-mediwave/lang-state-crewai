from datetime import datetime
from langchain_core.tools import tool
from langgraph.prebuilt import ToolExecutor
from langchain_community.utilities import OpenWeatherMapAPIWrapper
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import StructuredTool, tool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.pydantic_v1 import BaseModel, Field
import os


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
    
    weather = OpenWeatherMapAPIWrapper(openweathermap_api_key=os.environ['OPENWEATHERMAP_API_KEY'])
    weather_data = weather.run(location)
    return weather_data


weather = StructuredTool.from_function(
    func=get_weather,
    name='Weather',
    description="useful for getting the current weather data about a partidular given location",
    handle_tool_error=True
)

# -------------------  WIKIPEDIA TOOL --------------


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
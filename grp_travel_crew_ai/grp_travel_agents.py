from crewai import Agent 
from textwrap import dedent
from langchain_community.chat_models import ChatOllama 
from langchain_community.llms.ollama import Ollama

from dotenv import load_dotenv
load_dotenv()

import os 
from crewai import Agent
from textwrap import dedent
from langchain_openai import ChatOpenAI

from .grp_travel_tools import SearchTools
from .grp_travel_tools import CalculatorTools

"""
Creating Agents Cheat Sheet:
- Think like a boss. Work backwards from the goal and think which employee 
    you need to hire to get the job done.
- Define the Captain of the crew who orient the other agents towards the goal. 
- Define which experts the captain needs to communicate with and delegate tasks to.
    Build a top down structure of the crew.

Goal:
- Create a 7-day travel itinerary with detailed per-day plans,
    including budget, packing suggestions, and safety tips.

Captain/Manager/Boss:
- Expert Travel Agent

Employees/Experts to hire:
- City Selection Expert 
- Local Tour Guide


Notes:
- Agents should be results driven and have a clear goal in mind
- Role is their job title
- Goals should actionable
- Backstory should be their resume
"""


class TravelAgents:
    def __init__(self):
       self.llm = ChatOllama(model= os.environ['LLM'])

    def expert_travel_agent(self):
        return Agent(
            role="Expert Travel Agent",
            backstory=dedent(
                f"""Expert in travel planning and logistics. 
                I have decades of expereince making travel iteneraries."""),
            goal=dedent(f"""
                        Create a 7-day travel itinerary with detailed per-day plans,
                        include budget, packing suggestions, and safety tips.
                        """),
            tools=[
                SearchTools.search_internet,
                CalculatorTools.calculate
            ],
            verbose=True,
            llm=self.llm,
        )

    def city_selection_expert(self):
        return Agent(
            role="City Selection Expert",
            backstory=dedent(
                f"""Expert at analyzing travel data to pick ideal destinations"""),
            goal=dedent(
                f"""Select the best cities based on weather, season, prices, and traveler interests"""),
            tools=[SearchTools.search_internet],
            verbose=True,
            llm=self.llm,
        )

    def local_tour_guide(self):
        return Agent(
            role="Local Tour Guide",
            backstory=dedent(f"""Knowledgeable local guide with extensive information
        about the city, it's attractions and customs"""),
            goal=dedent(
                f"""Provide the BEST insights about the selected city"""),
            tools=[SearchTools.search_internet],
            verbose=True,
            llm=self.llm,
        )
    
    def travel_manager(self):
        return Agent(
            role="Travel agency manager",
            backstory=dedent(f"""Knowledgeable travel agency manager, orchestrating the tasks to the fellow crew members to obtain the tailored travel solutions to the user. """),
            goal=dedent(
                f"""orchestrate the tasks to the crew members and supervise them, find the need of the user"""),
            verbose=True,
            llm=self.llm,
        )

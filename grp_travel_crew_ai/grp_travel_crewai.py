from crewai import Crew
from textwrap import dedent
# from agents import TravelAgents
from crewai.process import Process
from dotenv import load_dotenv
load_dotenv()
import os

from crewai import Task
from textwrap import dedent
from langchain_community.llms.ollama import Ollama
from langchain_community.chat_models.ollama import ChatOllama
from langchain_core.messages import AIMessage

from .grp_travel_agents import TravelAgents
from .grp_travel_task import TravelTask


def travel_crew(state):
    
    input = state['messages'][0].content
    
    
    travel_agents = TravelAgents()
    travel_tasks = TravelTask()
    
    llm = ChatOllama(model=os.environ['LLM'])
    
    expert_travel_agent = travel_agents.expert_travel_agent()
    city_selection_agent = travel_agents.city_selection_expert()
    local_tour_guide = travel_agents.local_tour_guide()
    travel_manager = travel_agents.travel_manager()
    
    
    travel_task_ = travel_tasks.travel_task(
        agent=travel_manager,
        input = input
    )
    


    crew = Crew(
        agents=[
            expert_travel_agent,
            city_selection_agent,
            local_tour_guide,
            travel_manager
        ],
        tasks=[
            travel_task_
        ],
        verbose=0,
        process=Process.hierarchical,
        # manager_callbacks=travel_manager,
        manager_llm=llm
    )
    
    result = crew.kickoff()
    print(result, end='\n\n')
    return {"messages": [AIMessage(content=result)], 'next': 'supervisor'}








































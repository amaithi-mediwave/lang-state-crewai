from crewai import Crew
from crewai.process import Process
import os
from langchain_groq.chat_models import ChatGroq
from langchain_core.agents import AgentFinish

from .grp_food_agents import ChefAgents
from .grp_food_task import ChefTask


def food_crew(input):
    
    agents = ChefAgents()
    tasks = ChefTask()
    
    LLM = os.getenv('LLM')
    llm = ChatGroq(
                model=os.environ['LLM'],
                api_key=os.environ['GROQ_API_KEY']
                )

    chef_agent = agents.chef_agent()
    nutrition_agent = agents.nutrition_agent()
    ingridient_agent = agents.ingridient_agent()
    image_visualizer_agent = agents.image_visualizer_agent()
    grocery_agent = agents.grocery_agent()
    wine_agent = agents.wine_agent()
    trivia_agent = agents.trivia_agent()
    manager_agent = agents.manager_agent()


    task1 = tasks.chef_task(
            agent=manager_agent,
            input=input,
            )


    crew = Crew(
        agents=[chef_agent,
                nutrition_agent,
                ingridient_agent,
                image_visualizer_agent,
                grocery_agent,
                wine_agent,
                trivia_agent,
                manager_agent
                ],
        tasks=[
                task1
            
        ],
        verbose=1,
        process=Process.hierarchical,
        # manager_callbacks=manager_agent,
        manager_llm=llm
    )

    result = crew.kickoff()

    return {"agent_outcome": AgentFinish(return_values={'output': result}, log=result)}

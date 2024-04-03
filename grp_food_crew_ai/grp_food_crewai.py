from crewai import Crew
from crewai.process import Process
import os
from langchain_community.llms.ollama import Ollama
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain_core.agents import AgentFinish

from .grp_food_agents import ChefAgents
from .grp_food_task import ChefTask


def food_crew(input):
    
    agents = ChefAgents()
    tasks = ChefTask()
    
    LLM = os.getenv('LLM')
    llm = Ollama(model=LLM)
    # function_llm = OllamaFunctions(model=LLM)

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
            # tasks.crew_task(input='how to make chicken biryani'),
            
            # tasks.ingriedient_finder
        ],
        # function_calling_llm=function_llm,
        # full_output=True,
        verbose=1,
        process=Process.hierarchical,
        # manager_callbacks=manager_agent,
        manager_llm=llm
    )

    result = crew.kickoff()
    # print(result)
    # return {"messages": [result]}
    return {"agent_outcome": AgentFinish(return_values={'output': result}, log=result)}

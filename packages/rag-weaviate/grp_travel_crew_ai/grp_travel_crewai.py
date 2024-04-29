from crewai import Crew
from textwrap import dedent
from crewai.process import Process
from langchain_groq.chat_models import ChatGroq
from langchain_core.agents import AgentFinish
import os


from .grp_travel_agents import TravelAgents
from .grp_travel_task import TravelTask


def travel_crew(state):
    
    input = state['messages'][0].content
    
    
    travel_agents = TravelAgents()
    travel_tasks = TravelTask()
    
    llm = ChatGroq(
                model=os.environ['LLM'],
                api_key=os.environ['GROQ_API_KEY']
                )
    
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
    # print(result, end='\n\n')
    # return {"messages": [AIMessage(content=result)], 'next': 'supervisor'}

    return {
        "agent_outcome": AgentFinish(
            return_values={
                'output': result
                }, 
            log=result
            )
        }







































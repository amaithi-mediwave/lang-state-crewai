

import operator
from typing import Annotated, Any, Dict, List, Optional, Sequence, TypedDict
import functools

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage

from .supervisor import supervisor_node, members 





# The agent state is the input to each node in the graph
class AgentState(TypedDict):
    # The annotation tells the graph that new messages will always
    # be added to the current states
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # The 'next' field indicates where to route to next
    next: str


# research_agent = create_agent(llm, [tavily_tool], "You are a web researcher.")
# research_node = functools.partial(agent_node, agent=research_agent, name="Researcher")

# # NOTE: THIS PERFORMS ARBITRARY CODE EXECUTION. PROCEED WITH CAUTION
# code_agent = create_agent(
#     llm,
#     [python_repl_tool],
#     "You may generate safe python code to analyze data and generate charts using matplotlib.",
# )
# code_node = functools.partial(agent_node, agent=code_agent, name="Coder")

# members = ["Food_crew", "General_conversation", "General_other", "Mediwave_rag", "Travel_crew"]






# from grp_travel_crew_ai.grp_travel_crewai import travel_crew

from grp_RAG1.grp_rag1_rag import mediwave_rag

from grp_others.grp_others_graph import graph_others as gen_others

# from grp_food_crew_ai.grp_food_crewai import food_crew

from grp_Gen_Conv.grp_gen_conv_chain import general_conversation


def final_graph():

    workflow = StateGraph(AgentState)

    # workflow.add_node("Food_crew", food_crew)
    workflow.add_node("General_conv", general_conversation)
    workflow.add_node("General_other", gen_others)
    workflow.add_node("Mediwave_rag", mediwave_rag)
    # workflow.add_node("Travel_crew", travel_crew)

    workflow.add_node("supervisor", supervisor_node)


    for member in members:
        
        if member == 'Mediwave_rag':
            continue
        
        
        # We want our workers to ALWAYS "report back" to the supervisor when done
        workflow.add_edge(member, "supervisor")
        

    # The supervisor populates the "next" field in the graph state
    # which routes to a node or finishes


    conditional_map = {k: k for k in members}



    conditional_map["FINISH"] = END
    # conditional_map['supervisor'] ='supervisor'

    workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)

    # Finally, add entrypoint
    workflow.set_entry_point("supervisor")
    workflow.set_finish_point('Mediwave_rag')

    # graph = workflow.compile()

    graph = workflow.compile()
    
    return graph
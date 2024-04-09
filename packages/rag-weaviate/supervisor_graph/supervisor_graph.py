

import operator
from typing import Annotated, Sequence, TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage
from typing import Annotated, Sequence, TypedDict, Union
from langchain_core.agents import AgentAction, AgentFinish


from grp_travel_crew_ai.grp_travel_crewai import travel_crew
from grp_RAG1.grp_rag1_rag import mediwave_rag
from grp_others.grp_others_graph import grp_other_def as gen_others
from grp_food_crew_ai.grp_food_crewai import food_crew
from grp_Gen_Conv.grp_gen_conv_chain import general_conversation
from .supervisor import supervisor_node, members 



# The agent state is the input to each node in the graph
class AgentState(TypedDict):

    messages: Annotated[Sequence[BaseMessage], operator.add]
    
    agent_outcome: Union[AgentAction, AgentFinish, None, str]

    next: dict



workflow = StateGraph(AgentState)


workflow.add_node("Food_crew", food_crew)
workflow.add_node("General_conv", general_conversation)
workflow.add_node("General_other", gen_others)
workflow.add_node("Mediwave_rag", mediwave_rag)
workflow.add_node("Travel_crew", travel_crew)

workflow.add_node("supervisor", supervisor_node)




# for member in members:
    
#     if member == 'Mediwave_rag':
#         continue
#     if member == 'Travel_crew':
#         continue
    

# # We want our workers to ALWAYS "report back" to the supervisor when done
# workflow.add_edge(member, "supervisor")


# The supervisor populates the "next" field in the graph state
# which routes to a node or finishes


conditional_map = {k: k for k in members}



# conditional_map["FINISH"] = END
# conditional_map['supervisor'] ='supervisor'

workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)

# Finally, add entrypoint
workflow.set_entry_point("supervisor")
# workflow.set_finish_point('Mediwave_rag')
# workflow.set_finish_point('General_conv')
# workflow.set_finish_point('Travel_crew')

for member in members:
    
    workflow.set_finish_point(member)
    # if member == 'Mediwave_rag':
    #     continue
    # if member == 'Travel_crew':
    #     continue
    

# We want our workers to ALWAYS "report back" to the supervisor when done
# workflow.add_edge(member, "supervisor")


supervisor_graph = workflow.compile()


supervisor_graph
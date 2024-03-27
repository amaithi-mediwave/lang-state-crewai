from langgraph.graph import END, StateGraph
from langchain_core.messages import AIMessage


from .grp_others_state import AgentState
from .grp_others_nodes import run_agent, execute_tools, should_continue


workflow = StateGraph(AgentState)

workflow.add_node("agent", run_agent)
workflow.add_node("action", execute_tools)


workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent", should_continue, {"continue": "action", "end": END}
)


workflow.add_edge("action", "agent")

graph_others = workflow.compile()



def grp_other_def(state):
    input = state['messages'][0].content
    
    print(input)
    
    res = graph_others.invoke({'input': input})
    
    result = res['agent_outcome'].return_values['output']
    
    return {'messages': [AIMessage(content=result)]}
    
from langgraph.prebuilt import ToolExecutor, ToolInvocation

from .grp_others_tools import tool_executor
from .grp_others_state import agent_runnable

#--------------- Node : EXECUTE TOOLS_NODE -------------------------
def execute_tools(state):
    print("Called `execute_tools`")
    messages = [state["agent_outcome"]]
    last_message = messages[-1]

    tool_name = last_message.tool

    print(f"Calling tool: {tool_name}")

    action = ToolInvocation(
        tool=tool_name,
        tool_input=last_message.tool_input,
    )
    response = tool_executor.invoke(action)
    return {"intermediate_steps": [(state["agent_outcome"], response)]}


#--------------- Node : RUN_AGENT_NODE -------------------------

def run_agent(state):
    # """
    #if you want to better manages intermediate steps
    inputs = state.copy()
    if len(inputs['intermediate_steps']) > 5:
        inputs['intermediate_steps'] = inputs['intermediate_steps'][-5:]
    # """
    agent_outcome = agent_runnable.invoke(state)
    return {"agent_outcome": agent_outcome}


#--------------- Node : SHOULD_CONTINUE_NODE ---------------------
def should_continue(state):
    messages = [state["agent_outcome"]]
    last_message = messages[-1]
    if "Action" not in last_message.log:
        return "end"
    else:
        return "continue"
    
# ----------------------------------------------------------------


from plan_and_execute.state import PlanExecute 
from plan_and_execute.planner import planner
from plan_and_execute.re_plan import replanner 
from plan_and_execute.re_plan import Response
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import chain

from supervisor_graph.supervisor_graph import supervisor_graph
from langchain_core.messages import BaseMessage, AIMessage,  HumanMessage
from operator import itemgetter
from typing import List

from dotenv import load_dotenv
load_dotenv()


async def execute_step(super_state: PlanExecute):
    task = super_state["plan"][0]
    
    
    # agent_response = await agent_executor.ainvoke({"input": task, "chat_history": []})
    
    agent_response = await supervisor_graph.ainvoke({"messages": [
        HumanMessage(
            content=task
        )
    ],
        
    })
    
    print(agent_response)
    
    return {
        "past_steps": (task, agent_response["agent_outcome"].return_values["output"])
    }


async def plan_step(super_state: PlanExecute):
    plan = await planner.ainvoke({"objective": super_state["input"]})
    return {"plan": plan.steps}


async def replan_step(super_state: PlanExecute):
    output = await replanner.ainvoke(super_state)
    if isinstance(output, Response):
        return {"response": output.response}
    else:
        return {"plan": output.steps}


def should_end(super_state: PlanExecute):
    if super_state["response"]:
        return True
    else:
        return False





from langgraph.graph import StateGraph, END

workflow = StateGraph(PlanExecute)

# Add the plan node
workflow.add_node("planner", plan_step)

# Add the execution step
workflow.add_node("agent", execute_step)

# Add a replan node
workflow.add_node("replan", replan_step)

workflow.set_entry_point("planner")

# From plan we go to agent
workflow.add_edge("planner", "agent")

# From agent, we replan
workflow.add_edge("agent", "replan")

workflow.add_conditional_edges(
    "replan",
    # Next, we pass in the function that will determine which node is called next.
    should_end,
    {
        # If `tools`, then we call the tool node.
        True: END,
        False: "agent",
    },
)


# Add typing for input
class GraphModel(BaseModel):
    input: str




# Finally, we compile it!
graph = workflow.compile().with_config({"run_name": "Super Graph"}).with_types(input_type=GraphModel)

# graph = graph.with_types(input_type=GraphModel)



# @chain
# async def custom_chain(input):
    
#     # input = input.content
#     print(input)
    
#     result = await graph.ainvoke({"input": input})
    
#     # print(result)
    
#     return AIMessage(content=result['response'])


# custom_chain = custom_chain.with_types(input_type=GraphModel)


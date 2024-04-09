from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Tuple, Annotated, TypedDict
import operator

#   plan: List[str]

class PlanExecute(TypedDict):
    input: str
    plan: List[dict]
    past_steps: Annotated[List[Tuple], operator.add]
    response: str

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes
from plan_and_execute.graph import graph1 as rag_weaviate_chain


# graph = graph.graph
app = FastAPI()


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


# Edit this to add the chain you want to add
add_routes(app, rag_weaviate_chain, path="/rag-weaviate", playground_type='default', enable_feedback_endpoint=True,)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

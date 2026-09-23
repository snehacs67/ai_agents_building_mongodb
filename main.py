import os
import key_param
from pymongo import MongoClient
from langchain_core.tools import tool
from typing import List, Annotated
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import ToolMessage
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.mongodb import MongoDBSaver
import voyageai


def init_mongodb():
    """
    Initialize MongoDB client and retrieve vector and full document collections.
    """
    mongo_uri = os.getenv("MONGODB_URI")
    client = MongoClient(mongo_uri)
    db = client["ai_agents"]
    vs_collection = db["vector_search"]
    full_collection = db["full_docs"]
    return client, vs_collection, full_collection


@tool
def get_information_for_question_answering(question: str) -> str:
    """
    Performs vector search on MongoDB Atlas using Voyage AI embeddings to retrieve relevant documentation.
    """
    mongodb_client, vs_collection, full_collection = init_mongodb()

    vo = voyageai.Client(api_key=key_param.voyage_api_key)
    query_vector = vo.embed([question], model="voyage-3-lite").embeddings[0]

    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding",
                "queryVector": query_vector,
                "numCandidates": 100,
                "limit": 3
            }
        },
        {
            "$project": {
                "_id": 0,
                "text": 1,
                "score": {"$meta": "vectorSearchScore"}
            }
        }
    ]

    results = list(vs_collection.aggregate(pipeline))

    if not results:
        return "No relevant documentation found."

    return "\n\n".join([doc.get("text", "") for doc in results])


@tool
def get_page_content_for_summarization(doc_id: str) -> str:
    """
    Retrieves complete page content from MongoDB full documentation collection by document title or ID.
    """
    mongodb_client, vs_collection, full_collection = init_mongodb()

    doc = full_collection.find_one({"$or": [{"doc_id": doc_id}, {"title": doc_id}]})

    if not doc:
        return f"No document found matching: {doc_id}"

    return doc.get("body", doc.get("text", "No content available."))


# Define the graph state type with messages that can accumulate
class GraphState(TypedDict):
    messages: Annotated[list, add_messages]


def agent(state: GraphState, llm_with_tools) -> GraphState:
    """
    Agent node.

    Args:
        state (GraphState): The graph state.
        llm_with_tools: The LLM with tools.

    Returns:
        GraphState: The updated messages.
    """
    messages = state["messages"]
    result = llm_with_tools.invoke(messages)
    return {"messages": [result]}


def tool_node(state: GraphState, tools_by_name) -> GraphState:
    """
    Tool node.

    Args:
        state (GraphState): The graph state.
        tools_by_name (Dict[str, Callable]): The tools by name.

    Returns:
        GraphState: The updated messages.
    """
    result = []
    tool_calls = state["messages"][-1].tool_calls

    for tool_call in tool_calls:
        tool_func = tools_by_name[tool_call["name"]]
        observation = tool_func.invoke(tool_call["args"])
        result.append(ToolMessage(content=str(observation), tool_call_id=tool_call["id"]))

    return {"messages": result}


def route_tools(state: GraphState):
    """
    Route to the tool node if the last message has tool calls. Otherwise, route to the end.

    Args:
        state (GraphState): The graph state.

    Returns:
        str: The next node to route to.
    """
    messages = state.get("messages", [])

    if len(messages) > 0:
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")

    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"

    return END


def init_graph(llm_with_tools, tools_by_name, mongodb_client):
    """
    Initialize the graph with MongoDB checkpointer memory.

    Args:
        llm_with_tools: The LLM with tools.
        tools_by_name (Dict[str, Callable]): The tools by name.
        mongodb_client (MongoClient): The MongoDB client.

    Returns:
        StateGraph: The compiled graph with checkpointer.
    """
    graph = StateGraph(GraphState)

    graph.add_node("agent", lambda state: agent(state, llm_with_tools))
    graph.add_node("tools", lambda state: tool_node(state, tools_by_name))

    graph.add_edge(START, "agent")
    graph.add_edge("tools", "agent")
    graph.add_conditional_edges("agent", route_tools, {"tools": "tools", END: END})

    checkpointer = MongoDBSaver(mongodb_client)

    return graph.compile(checkpointer=checkpointer)


def execute_graph(app, thread_id: str, user_input: str) -> None:
    """
    Stream outputs from the graph using a thread_id for session memory.

    Args:
        app: The compiled graph application.
        thread_id (str): The thread ID.
        user_input (str): The user's input.
    """
    input_payload = {"messages": [("user", user_input)]}
    config = {"configurable": {"thread_id": thread_id}}

    for output in app.stream(input_payload, config):
        for key, value in output.items():
            print(f"Node {key}:")
            print(value)

    print("---FINAL ANSWER---")
    print(value["messages"][-1].content)


def main():
    """
    Main function to initialize and execute the graph with session memory.
    """
    mongodb_client, vs_collection, full_collection = init_mongodb()

    tools = [
        get_information_for_question_answering,
        get_page_content_for_summarization
    ]

    llm = ChatOpenAI(openai_api_key=key_param.openai_api_key, temperature=0, model="gpt-4o")

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful AI assistant."
                " You are provided with tools to answer questions and summarize technical documentation related to MongoDB."
                " Think step-by-step and use these tools to get the information required to answer the user query."
                " Do not re-run tools unless absolutely necessary."
                " If you are not able to get enough information using the tools, reply with I DON'T KNOW."
                " You have access to the following tools: {tool_names}."
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))

    bind_tools = llm.bind_tools(tools)
    llm_with_tools = prompt | bind_tools

    tools_by_name = {tool.name: tool for tool in tools}

    app = init_graph(llm_with_tools, tools_by_name, mongodb_client)

    print("\n=== Call 1 (Thread '1') ===")
    execute_graph(app, "1", "What are some best practices for data backups in MongoDB?")

    print("\n=== Call 2 (Thread '1' - Memory Check) ===")
    execute_graph(app, "1", "What did I just ask you?")


if __name__ == "__main__":
    main()
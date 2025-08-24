"""LangGraph single-node graph template.

Returns a predefined response. Replace logic and configuration as needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, TypedDict, Annotated
from operator import add

from langchain_core.messages import BaseMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph
from langgraph.runtime import Runtime
from langchain_openai import ChatOpenAI


class Context(TypedDict):
    """Context parameters for the agent.

    Set these when creating assistants OR when invoking the graph.
    See: https://langchain-ai.github.io/langgraph/cloud/how-tos/configuration_cloud/
    """

    model_name: str
    system_prompt: str


@dataclass
class State:
    """Input state for the agent.

    Defines the initial structure of incoming data.
    See: https://langchain-ai.github.io/langgraph/concepts/low_level/#state
    """

    messages: Annotated[List[BaseMessage], add]


async def call_model(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Process input and returns output.

    Can use runtime context to alter behavior.
    """
    # Get configuration
    model_name = 'gpt-4'
    system_prompt = 'keep all responses fewer than 100 words.'
    
    # Initialize the model
    model = ChatOpenAI(model=model_name)
    
    # Prepare messages for the LLM
    # Start with system message if this is the beginning of conversation
    messages_for_llm = []
    
    # Add system message if no previous messages or first message isn't system
    if not state.messages or not isinstance(state.messages[0], SystemMessage):
        messages_for_llm.append(SystemMessage(content=system_prompt))
    
    # Add all conversation history
    messages_for_llm.extend(state.messages)
    
    # Call the LLM with full conversation context
    response = await model.ainvoke(messages_for_llm)
    
    return {"messages": [response]}


# Define the graph
graph = (
    StateGraph(State, context_schema=Context)
    .add_node(call_model)
    .add_edge("__start__", "call_model")
    .compile(name="New Graph")
)

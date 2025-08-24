"""LangGraph single-node graph template.

Returns a predefined response. Replace logic and configuration as needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, TypedDict, Annotated
from operator import add

from langchain_core.messages import BaseMessage, AIMessage, SystemMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph
from langgraph.runtime import Runtime
from langchain_openai import ChatOpenAI
from langchain_community.tools import TavilySearchResults
from langchain_core.tools import tool
import json
import asyncio
import uuid


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
    needs_search: bool = False
    search_context: str = ""


async def preprocess_query(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Determine if the query needs web search."""
    if not state.messages:
        return {"needs_search": False}
    
    last_message = state.messages[-1]
    
    # Messages are always received as dict format
    message_type = last_message.get('type')
    content_data = last_message.get('content', [])
    
    # Extract text content from the content array
    if isinstance(content_data, list) and len(content_data) > 0:
        text_content = next((item['text'] for item in content_data if item.get('type') == 'text'), None)
        message_content = text_content
    else:
        message_content = str(content_data)
    
    if message_type != "human" or not message_content:
        return {"needs_search": False}
    
    # Use LLM to determine if search is needed
    model = ChatOpenAI(model='gpt-3.5-turbo', temperature=0)
    
    search_decision_prompt = SystemMessage(
        content="""You are a search decision agent. Determine if the user's query requires current/recent information from the web.

        Return ONLY "YES" if the query needs web search (for current events, recent news, today's information, live data, etc.)
        Return ONLY "NO" if the query can be answered with general knowledge.
        
        Examples that need search: "What's the weather today?", "Latest news about AI", "Current stock price", "Recent events"
        Examples that don't need search: "How to code in Python?", "Explain quantum physics", "What is photosynthesis?"
        """
    )
    
    # Create HumanMessage from extracted content
    human_message = HumanMessage(content=message_content)
    messages = [search_decision_prompt, human_message]
    response = await model.ainvoke(messages)
    
    needs_search = response.content.strip().upper() == "YES"
    
    return {"needs_search": needs_search}


async def web_search(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Perform web search and summarize results."""
    print("🌐 WEB_SEARCH: Starting search")
    
    if not state.messages:
        print("🌐 WEB_SEARCH: No messages, returning empty context")
        return {"search_context": ""}
    
    # Find the most recent human message for the search query
    human_message = None
    for msg in reversed(state.messages):
        if isinstance(msg, dict) and msg.get('type') == 'human':
            human_message = msg
            break
        elif hasattr(msg, 'type') and msg.type == 'human':
            human_message = msg
            break
    
    if not human_message:
        print("🌐 WEB_SEARCH: No human message found")
        return {"search_context": ""}
    
    # Extract content from both dict and HumanMessage formats
    if isinstance(human_message, dict):
        content_data = human_message.get('content', [])
        if isinstance(content_data, list) and len(content_data) > 0:
            query = next((item['text'] for item in content_data if item.get('type') == 'text'), None)
        else:
            query = str(content_data)
    elif hasattr(human_message, 'content'):
        query = human_message.content
    else:
        query = str(human_message)
    
    print(f"🌐 WEB_SEARCH: Query: '{query}'")
    
    # Create tool call ID for single comprehensive message
    tool_call_id = str(uuid.uuid4())
    
    # Perform web search
    try:
        search_tool = TavilySearchResults(max_results=5)
        # Use asyncio.to_thread to run blocking operation in separate thread
        search_results = await asyncio.to_thread(search_tool.invoke, query)
        print(f"🌐 WEB_SEARCH: Got {len(search_results)} results")
        
    except Exception as e:
        print(f"🌐 WEB_SEARCH: Search failed: {e}")
        error_message = ToolMessage(
            content=f"🔍 **Web Search**\n\n*Query: {query}*\n\n❌ Search failed: {str(e)}",
            tool_call_id=tool_call_id
        )
        return {
            "search_context": f"Web search failed: {str(e)}",
            "messages": [error_message]
        }
    
    # Summarize search results
    model = ChatOpenAI(model='gpt-3.5-turbo', temperature=0)
    
    summarize_prompt = SystemMessage(
        content="""Summarize the following search results into key facts relevant to the user's query. 
        Be concise but comprehensive. Focus on the most important and recent information.
        Format as bullet points."""
    )
    
    search_content = json.dumps(search_results, indent=2)
    summarize_message = HumanMessage(
        content=f"User query: {query}\n\nSearch results:\n{search_content}"
    )
    
    summary_response = await model.ainvoke([summarize_prompt, summarize_message])
    
    # Create single comprehensive tool message
    comprehensive_message = ToolMessage(
        content=f"🔍 **Web Search**\n\n*Query: {query}*\n\n✅ Found {len(search_results)} results\n\n**Summary:**\n{summary_response.content}",
        tool_call_id=tool_call_id
    )
    
    return {
        "search_context": summary_response.content,
        "messages": [comprehensive_message]
    }


async def call_model(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Process input and returns output.

    Can use runtime context to alter behavior.
    """
    # Get configuration
    model_name = 'gpt-4'
    system_prompt = """keep all responses fewer than 100 words. If the prompt requires current information, 
    use the search context to answer the question. If the prompt does not require current information,
    answer the question with general knowledge."""

    # Initialize the model
    model = ChatOpenAI(model=model_name)
    
    # Prepare messages for the LLM
    # Start with system message if this is the beginning of conversation
    messages_for_llm = []
    
    # Add system message if no previous messages or first message isn't system
    enhanced_system_prompt = system_prompt
    if state.search_context:
        enhanced_system_prompt += f"\n\nAdditional context from web search:\n{state.search_context}"
    
    if not state.messages or not isinstance(state.messages[0], SystemMessage):
        messages_for_llm.append(SystemMessage(content=enhanced_system_prompt))
    
    # Add conversation history, but filter out ToolMessages (they're for UI only)
    filtered_messages = [msg for msg in state.messages if not isinstance(msg, ToolMessage)]
    messages_for_llm.extend(filtered_messages)
    
    # Call the LLM with full conversation context
    response = await model.ainvoke(messages_for_llm)
    
    return {"messages": [response]}


def should_search(state: State) -> str:
    """Determine the next node based on search decision."""
    return "web_search" if state.needs_search else "call_model"


# Define the graph
graph = (
    StateGraph(State, context_schema=Context)
    .add_node("preprocess", preprocess_query)
    .add_node("web_search", web_search)
    .add_node("call_model", call_model)
    .add_edge("__start__", "preprocess")
    .add_conditional_edges(
        "preprocess",
        should_search,
        {
            "web_search": "web_search",
            "call_model": "call_model"
        }
    )
    .add_edge("web_search", "call_model")
    .add_edge("call_model", "__end__")
    .compile(name="Search-Enhanced Chat Graph")
)

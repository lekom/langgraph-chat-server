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

    current_date = get_current_date_string()

    search_decision_prompt = SystemMessage(
        content=f"""You are a search decision agent. Determine if the user's query requires current/recent information from the web.

        Return ONLY "YES" if the query needs web search (for current events, recent news, today's information, live data, etc.)
        Return ONLY "NO" if the query can be answered with general knowledge.
        
        Examples that need search: "What's the weather today?", "Latest news about AI", "Current stock price", "Recent events"
        Examples that don't need search: "How to code in Python?", "Explain quantum physics", "What is photosynthesis?"

        The current date is {current_date}.  So there is no need to search for the current date.
        """
    )
    
    # Create HumanMessage from extracted content
    human_message = HumanMessage(content=message_content)
    messages = [search_decision_prompt, human_message]
    response = await model.ainvoke(messages)
    
    needs_search = response.content.strip().upper() == "YES"
    
    return {"needs_search": needs_search}


async def generate_improved_query(state: State, original_query: str) -> str:
    """Generate an improved search query based on conversation context."""
    # Get conversation context (excluding tool messages)
    context_messages = [msg for msg in state.messages if not isinstance(msg, ToolMessage)]
    
    if len(context_messages) <= 1:
        # No prior context, return original query
        return original_query
    
    # Build context string from entire conversation
    context_parts = []
    for msg in context_messages:  # Use entire conversation history
        if isinstance(msg, dict):
            msg_type = msg.get('type', 'unknown')
            if msg_type == 'human':
                content_data = msg.get('content', [])
                if isinstance(content_data, list) and len(content_data) > 0:
                    content = next((item['text'] for item in content_data if item.get('type') == 'text'), "")
                else:
                    content = str(content_data)
                context_parts.append(f"Human: {content}")
            elif msg_type == 'ai':
                content_data = msg.get('content', [])
                if isinstance(content_data, list) and len(content_data) > 0:
                    content = next((item['text'] for item in content_data if item.get('type') == 'text'), "")
                else:
                    content = str(content_data)
                context_parts.append(f"AI: {content}")
        elif hasattr(msg, 'content'):
            if hasattr(msg, 'type'):
                context_parts.append(f"{msg.type}: {msg.content}")
            else:
                context_parts.append(f"Message: {msg.content}")
    
    context_string = "\n".join(context_parts)
    
    # Use LLM to generate improved query
    model = ChatOpenAI(model='gpt-3.5-turbo', temperature=0)
    
    current_date = get_current_date_string()

    improvement_prompt = SystemMessage(
        content=f"""You are a search query optimizer. Given a conversation context and a user's search query, generate a more specific and effective web search query.

        Guidelines:
        - If the query refers to pronouns (she, he, they, it), replace with specific names/entities from context
        - Add relevant context terms that would improve search results
        - Keep the query concise but specific
        - Focus on searchable facts and proper nouns
        - If no context is relevant, return the original query

        The current date is {current_date}.

        Examples:
        Context: "Tell me about Elizabeth Holmes, the tech entrepreneur"
        Query: "has she ever been to jail"
        Improved: "Elizabeth Holmes criminal history arrest record"

        Context: "We were discussing the iPhone 15"
        Query: "what are the specs"
        Improved: "iPhone 15 technical specifications features"

        Return ONLY the improved search query, no explanation.
        """
    )
    
    query_message = HumanMessage(
        content=f"Context:\n{context_string}\n\nUser's query: {original_query}\n\nImproved query:"
    )
    
    response = await model.ainvoke([improvement_prompt, query_message])
    improved_query = response.content.strip()
    
    # Fallback to original if improvement is empty or too similar
    if not improved_query or improved_query.lower() == original_query.lower():
        return original_query
    
    return improved_query


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
    
    print(f"🌐 WEB_SEARCH: Original query: '{query}'")
    
    # Generate improved search query using conversation context
    improved_query = await generate_improved_query(state, query)
    print(f"🌐 WEB_SEARCH: Improved query: '{improved_query}'")
    
    # Create tool call ID for single comprehensive message
    tool_call_id = str(uuid.uuid4())
    
    # Perform web search
    try:
        search_tool = TavilySearchResults(max_results=5)
        # Use asyncio.to_thread to run blocking operation in separate thread
        search_results = await asyncio.to_thread(search_tool.invoke, improved_query)
        print(f"🌐 WEB_SEARCH: Got {len(search_results)} results")
        
    except Exception as e:
        print(f"🌐 WEB_SEARCH: Search failed: {e}")
        error_message = ToolMessage(
            content=f"🔍 Web Search\n\nOriginal: {query}\nImproved: {improved_query}\n\n❌ Search failed: {str(e)}",
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
        content=f"Original query: {query}\nImproved query: {improved_query}\n\nSearch results:\n{search_content}"
    )
    
    summary_response = await model.ainvoke([summarize_prompt, summarize_message])
    
    # Create single comprehensive tool message
    comprehensive_message = ToolMessage(
        content=f"🔍 Web Search: {improved_query}  ✅ Found {len(search_results)} results.  Summary: {summary_response.content}",
        tool_call_id=tool_call_id
    )
    
    return {
        "search_context": summary_response.content,
        "messages": [comprehensive_message]
    }


def get_current_date_string() -> str:
    """Get current date and time in user's timezone formatted as 'Full_month day, year, current time in user's time zone'."""
    from datetime import datetime
    
    # Get user's local timezone
    local_tz = datetime.now().astimezone().tzinfo
    current_datetime = datetime.now(local_tz)
    
    # Format: "Full_month day, year, current time in user's time zone"
    return current_datetime.strftime("%B %d, %Y, %I:%M %p %Z")


async def call_model(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Process input and returns output.

    Can use runtime context to alter behavior.
    """
    # Get configuration
    model_name = 'gpt-4'
    current_date = get_current_date_string()
    
    system_prompt = f"""keep all responses fewer than 100 words. If the prompt requires current information, 
    use the search context to answer the question. If the prompt does not require current information,
    answer the question with general knowledge.  The current date is {current_date}."""

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

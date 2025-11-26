import json
import os

from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage ,SystemMessage
from langchain_core.runnables import RunnableConfig

from src.config.logger import get_logger
from src.llms.llm import get_llm_by_type
from src.tools import (
    crawl_tool,
    get_web_search_tool,
    python_repl_tool,
)
from src.tools.search import LoggedTavilySearch

logger = get_logger(__name__)




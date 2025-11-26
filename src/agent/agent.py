from src.config.logger import get_logger
from src.llms.llm import get_llm_by_type
from src.tools import (
    crawl_tool,
    get_web_search_tool,
    python_repl_tool,
)

from langchain.agents import create_agent
logger = get_logger(__name__)
llm = get_llm_by_type("basic")
 
agent = create_agent(
    model=llm,
    tools=[get_web_search_tool(),crawl_tool,python_repl_tool],
    system_prompt="You are a helpful assistant",
)

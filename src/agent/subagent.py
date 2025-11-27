from src.llms.llm import get_llm_by_type
from src.tools import (
    crawl_tool,
    get_web_search_tool,
    python_repl_tool,
    think_tool,
)
from src.middleware.debug_middleware import ToolMonitoringMiddleware, LoggingMiddleware
from src.prompts.prompts_zh import (
    RESEARCHER_INSTRUCTIONS,
    SUBAGENT_CODER_INSTRUCTIONS,
)

from datetime import datetime

research_sub_agent = {
    "name": "research-agent",
    "description": "将研究工作委托给副代理研究员。每次只给这个研究者一个课题。",
    "system_prompt": RESEARCHER_INSTRUCTIONS.format(date=datetime.now().strftime("%Y-%m-%d")),
    "tools": [get_web_search_tool(),think_tool, crawl_tool,python_repl_tool],
    "model": get_llm_by_type("reasoning"),
    #"middleware": [LoggingMiddleware()],
    "debug": False,
}

coder_sub_agent = {
    "name": "coder-agent",
    "description": "根据委托的任务，使用Python代码实现并执行。",
    "system_prompt": SUBAGENT_CODER_INSTRUCTIONS,
    "tools": [think_tool, python_repl_tool],
    "model": get_llm_by_type("code"),
    #"middleware": [LoggingMiddleware()],
    "debug": False,
}
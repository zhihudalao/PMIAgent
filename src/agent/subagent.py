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
from src.prompts.template import get_prompt_template

from datetime import datetime

research_sub_agent = {
    "name": "research-agent",
    "description": "将研究工作委托给子代理研究员。每次只给这个研究者一个课题。",
    "system_prompt": get_prompt_template("researcher"),
    "tools": [get_web_search_tool(),think_tool, crawl_tool],
    "model": get_llm_by_type("reasoning"),
    #"middleware": [LoggingMiddleware()],
    "debug": True,
}

coder_sub_agent = {
    "name": "coder-agent",
    "description": "根据委托的任务，使用Python代码实现并执行。",
    "system_prompt": get_prompt_template("coder"),
    "tools": [think_tool, python_repl_tool],
    "model": get_llm_by_type("code"),
    #"middleware": [LoggingMiddleware()],
    "debug": False,
}

researcher_sub_agent = {
    "name": "researcher-agent",
    "description": "将研究工作委托给副代理研究员。每次只给这个研究者一个课题。",
    "system_prompt": RESEARCHER_INSTRUCTIONS.format(date=datetime.now().strftime("%Y-%m-%d")),
    "tools": [get_web_search_tool(),think_tool, crawl_tool],
    "model": get_llm_by_type("reasoning"),
    #"middleware": [LoggingMiddleware()],
    "debug": False,
}



report_sub_agent = {
    "name": "report-agent",
    "description": "根据研究结果，生成报告。",
    "system_prompt": "根据研究结果，生成报告。",
    "tools": [think_tool],
    "model": get_llm_by_type("reasoning"),
    #"middleware": [LoggingMiddleware()],
    "debug": False,
}

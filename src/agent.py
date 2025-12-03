from datetime import datetime

from src.config.logger import get_logger
from src.llms.llm import get_llm_by_type
from src.prompts.template import get_prompt_template
from src.agent.agent import create_deepagent
from src.tools import (
    crawl_tool,
    get_web_search_tool,
    think_tool,
)
logger = get_logger(__name__)
# Limits
max_concurrent_research_units = 3
max_researcher_iterations = 3


# Combine orchestrator instructions (RESEARCHER_INSTRUCTIONS only for sub-agents)
INSTRUCTIONS = get_prompt_template("coordinator").format(
        CURRENT_TIME = datetime.now().strftime("%Y-%m-%d"),
        max_concurrent_research_units=max_concurrent_research_units,
        max_researcher_iterations=max_researcher_iterations,
    )

planer_sub_agent = {
    "name": "planer-agent",
    "description": "根据用户需求，规划研究计划。",
    "system_prompt": get_prompt_template("planner"),
    "tools": [get_web_search_tool()],
    "model": get_llm_by_type("reasoning"),
    #"middleware": [LoggingMiddleware()],
    "debug": True,
}

research_sub_agent = {
    "name": "research-agent",
    "description": "将研究工作委托给子代理研究员。每次只给这个研究者一个课题。",
    "system_prompt": get_prompt_template("researcher"),
    "tools": [get_web_search_tool(),think_tool, crawl_tool],
    "model": get_llm_by_type("reasoning"),
    #"middleware": [LoggingMiddleware()],
    "debug": True,
}



agent = create_deepagent(
    model=get_llm_by_type("basic"),
    tools=[],
    system_prompt=INSTRUCTIONS,
    subagents=[research_sub_agent],
    debug=True,
)


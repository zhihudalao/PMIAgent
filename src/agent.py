from src.config.logger import get_logger
from src.llms.llm import get_llm_by_type
from src.tools import (
    crawl_tool,
    get_web_search_tool,
    python_repl_tool,
    think_tool,
)
logger = get_logger(__name__)

from src.agent.agent import create_deepagent
from datetime import datetime

from src.prompts.prompts_zh import (
    RESEARCHER_INSTRUCTIONS,
    RESEARCH_WORKFLOW_INSTRUCTIONS,
    SUBAGENT_DELEGATION_INSTRUCTIONS,
)
# Limits
max_concurrent_research_units = 3
max_researcher_iterations = 3

# Get current date
current_date = datetime.now().strftime("%Y-%m-%d")

# Combine orchestrator instructions (RESEARCHER_INSTRUCTIONS only for sub-agents)
INSTRUCTIONS = (
    RESEARCH_WORKFLOW_INSTRUCTIONS
    + "\n\n"
    + "=" * 80
    + "\n\n"
    + SUBAGENT_DELEGATION_INSTRUCTIONS.format(
        max_concurrent_research_units=max_concurrent_research_units,
        max_researcher_iterations=max_researcher_iterations,
    )
)

base_model = get_llm_by_type("basic")
reasoning_model = get_llm_by_type("reasoning")


research_sub_agent = {
    "name": "research-agent",
    "description": "将研究工作委托给副代理研究员。每次只给这个研究者一个课题。",
    "system_prompt": RESEARCHER_INSTRUCTIONS.format(date=current_date),
    "tools": [get_web_search_tool(),think_tool, crawl_tool,python_repl_tool],
    "model": reasoning_model,
}

agent = create_deepagent(
    model=reasoning_model,
    tools=[get_web_search_tool(), think_tool,crawl_tool],
    system_prompt=INSTRUCTIONS,
    subagents=[research_sub_agent],
    #backend=FilesystemBackend(root_dir="./reports", virtual_mode=True),
)


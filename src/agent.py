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
from src.agent.subagent import research_sub_agent
from src.prompts.prompts_zh import (
    RESEARCH_WORKFLOW_INSTRUCTIONS,
    SUBAGENT_DELEGATION_INSTRUCTIONS,
)

# Limits
max_concurrent_research_units = 3
max_researcher_iterations = 3


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


agent = create_deepagent(
    model=get_llm_by_type("reasoning"),
    tools=[get_web_search_tool(), think_tool,crawl_tool],
    system_prompt=INSTRUCTIONS,
    subagents=[research_sub_agent]
)


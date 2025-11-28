from src.config.logger import get_logger
from src.llms.llm import get_llm_by_type

logger = get_logger(__name__)
from src.prompts.template import get_prompt_template
from src.agent.agent import create_deepagent
from src.agent.subagent import research_sub_agent


# Limits
max_concurrent_research_units = 3
max_researcher_iterations = 3


# Combine orchestrator instructions (RESEARCHER_INSTRUCTIONS only for sub-agents)
INSTRUCTIONS = get_prompt_template("coordinator").format(
        max_concurrent_research_units=max_concurrent_research_units,
        max_researcher_iterations=max_researcher_iterations,
    )


agent = create_deepagent(
    model=get_llm_by_type("basic"),
    tools=[],
    system_prompt=INSTRUCTIONS,
    subagents=[research_sub_agent],
    debug=True,
)


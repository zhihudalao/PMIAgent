"""Deepagents come with planning, filesystem, and subagents."""

from collections.abc import Callable, Sequence
from typing import Any
from pathlib import Path
import uuid

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware, InterruptOnConfig, TodoListMiddleware
from langchain.agents.middleware.summarization import SummarizationMiddleware
from langchain.agents.middleware.types import AgentMiddleware
from langchain.agents.structured_output import ResponseFormat
from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.cache.base import BaseCache
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore
from langgraph.types import Checkpointer

from deepagents.backends.protocol import BackendFactory, BackendProtocol
from deepagents.backends import CompositeBackend
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.middleware.patch_tool_calls import PatchToolCallsMiddleware
from deepagents.middleware.subagents import CompiledSubAgent, SubAgent, SubAgentMiddleware

from src.middleware.agent_memory import AgentMemoryMiddleware
from src.middleware.context_enhancement import ContextEnhancementMiddleware
from src.middleware.layered_memory import LayeredMemoryMiddleware
from src.middleware.logging import LoggingMiddleware
from src.middleware.memory_adapter import MemoryMiddlewareFactory
from src.middleware.performance_monitor import PerformanceMonitorMiddleware
from src.middleware.security import SecurityMiddleware

from src.llms.llm import get_llm_by_type
from src.config.logger import get_logger
logger = get_logger(__name__)

def create_deepagent(
    model: str | BaseChatModel | None = None,
    tools: Sequence[BaseTool | Callable | dict[str, Any]] | None = None,
    *,
    system_prompt: str | None = None,
    subagents: list[SubAgent | CompiledSubAgent] | None = None,
    response_format: ResponseFormat | None = None,
    context_schema: type[Any] | None = None,
    checkpointer: Checkpointer | None = None,
    store: BaseStore | None = None,
    interrupt_on: dict[str, bool | InterruptOnConfig] | None = None,
    debug: bool = False,
    name: str | None = None,
    cache: BaseCache | None = None,
) -> CompiledStateGraph:
    """Create a deep agent.

    This agent will by default have access to a tool to write todos (write_todos),
    seven file and execution tools: ls, read_file, write_file, edit_file, glob, grep, execute,
    and a tool to call subagents.

    The execute tool allows running shell commands if the backend implements SandboxBackendProtocol.
    For non-sandbox backends, the execute tool will return an error message.

    Args:
        model: The model to use. Defaults to Claude Sonnet 4.
        tools: The tools the agent should have access to.
        system_prompt: The additional instructions the agent should have. Will go in
            the system prompt.
        middleware: Additional middleware to apply after standard middleware.
        subagents: The subagents to use. Each subagent should be a dictionary with the
            following keys:
                - `name`
                - `description` (used by the main agent to decide whether to call the
                  sub agent)
                - `prompt` (used as the system prompt in the subagent)
                - (optional) `tools`
                - (optional) `model` (either a LanguageModelLike instance or dict
                  settings)
                - (optional) `middleware` (list of AgentMiddleware)
        response_format: A structured output response format to use for the agent.
        context_schema: The schema of the deep agent.
        checkpointer: Optional checkpointer for persisting agent state between runs.
        store: Optional store for persistent storage (required if backend uses StoreBackend).
        backend: Optional backend for file storage and execution. Pass either a Backend instance
            or a callable factory like `lambda rt: StateBackend(rt)`. For execution support,
            use a backend that implements SandboxBackendProtocol.
        interrupt_on: Optional Dict[str, bool | InterruptOnConfig] mapping tool names to
            interrupt configs.
        debug: Whether to enable debug mode. Passed through to create_agent.
        name: The name of the agent. Passed through to create_agent.
        cache: The cache to use for the agent. Passed through to create_agent.

    Returns:
        A configured deep agent.
    """
    if model is None:
        model = get_llm_by_type("basic")

    # # 长期记忆目录, 指向 ./.deepagents/AGENT_NAME/ with /memories/ prefix
    # assistant_id = "deepagent"
    # # 长期记忆目录, 指向 ./.deepagents/AGENT_NAME/ with /memories/ prefix
    # agent_dir = Path.cwd() / ".deepagents" / assistant_id
    # agent_dir.mkdir(parents=True, exist_ok=True)
    # # agent_md = agent_dir / "agent.md"
    # # if not agent_md.exists():
    # #     source_content = get_default_coding_instructions()
    # #     agent_md.write_text(source_content)
    #
    # # 长期记忆后端 - rooted at agent directory
    # # 处理 /memories/ files 和 /agent.md
    # # virtual_mode放置路径遍历攻击
    # long_term_backend = FilesystemBackend(root_dir=agent_dir, virtual_mode=True)
    #
    # # Composite backend: current working directory for default, agent directory for /memories/
    # backend = CompositeBackend(
    #     default=FilesystemBackend(), routes={"/memories/": long_term_backend}
    # )
    # # 建中间件管道
    # agent_middleware = []
    # logger.info("[bold blue]🏗️  正在构建中间件管道系统...[/bold blue]")
    #
    # # 第一层：全局监控（最外层）
    # # 1. 性能监控中间件
    # try:
    #     performance_middleware = PerformanceMonitorMiddleware(
    #         backend=long_term_backend,
    #         metrics_path="/performance/",
    #         enable_system_monitoring=True,
    #         max_records=1000,
    #     )
    #     agent_middleware.append(performance_middleware)
    #     logger.info("[green]✓[/green] 性能监控中间件 (最外层)")
    # except Exception as e:
    #     logger.warning(f"[yellow]⚠ 性能监控中间件失败: {e}[/yellow]")
    #
    # # 2. 日志记录中间件
    # try:
    #     logging_middleware = LoggingMiddleware(
    #         backend=long_term_backend,
    #         session_id=assistant_id,
    #         log_path="/logs/",
    #         enable_conversation_logging=True,
    #         enable_tool_logging=True,
    #         enable_performance_logging=True,
    #         enable_error_logging=True,
    #     )
    #     agent_middleware.append(logging_middleware)
    #     logger.info("[green]✓[/green] 日志记录中间件")
    # except Exception as e:
    #     logger.warning(f"[yellow]⚠ 日志记录中间件失败: {e}[/yellow]")
    #
    # # 第二层：上下文增强
    # # 3. 上下文增强中间件
    # try:
    #     context_middleware = ContextEnhancementMiddleware(
    #         backend=long_term_backend,
    #         context_path="/context/",
    #         enable_project_analysis=True,
    #         enable_user_preferences=True,
    #         enable_conversation_enhancement=True,
    #         max_context_length=4000,
    #     )
    #     agent_middleware.append(context_middleware)
    #     logger.info("[green]✓[/green] 上下文增强中间件")
    # except Exception as e:
    #     logger.warning(f"[yellow]⚠ 上下文增强中间件失败: {e}[/yellow]")
    #
    # # 4. 分层记忆中间件（在上下文增强之后，框架之前）
    # try:
    #     memory_middleware = MemoryMiddlewareFactory.auto_upgrade_memory(
    #         backend=long_term_backend,
    #         memory_path="/memories/",
    #         enable_layered=None,  # 自动检测
    #         working_memory_size=10,
    #         enable_semantic_memory=True,
    #         enable_episodic_memory=True,
    #     )
    #
    #     if isinstance(memory_middleware, list):
    #         # 混合模式，返回多个中间件
    #         agent_middleware.extend(memory_middleware)
    #         logger.info("[green]✓[/green] 分层记忆系统 (混合模式)")
    #     else:
    #         # 单个中间件
    #         agent_middleware.append(memory_middleware)
    #         if hasattr(memory_middleware, "__class__"):
    #             if isinstance(memory_middleware, LayeredMemoryMiddleware):
    #                 logger.info("[green]✓[/green] 分层记忆系统")
    #             elif isinstance(memory_middleware, AgentMemoryMiddleware):
    #                 logger.info("[green]✓[/green] 传统记忆系统")
    #
    # except Exception as e:
    #     # 如果分层记忆失败，回退到传统记忆
    #     logger.warning(f"[yellow]⚠ 记忆系统失败，使用传统模式: {e}[/yellow]")
    #     agent_middleware.append(
    #         AgentMemoryMiddleware(backend=long_term_backend, memory_path="/memories/")
    #     )


    deepagent_middleware = [
        TodoListMiddleware(),
        #FilesystemMiddleware(backend=backend),
        SubAgentMiddleware(
            default_model=model,
            default_tools=tools,
            subagents=subagents if subagents is not None else [],
            default_middleware=[
                TodoListMiddleware(),
                #FilesystemMiddleware(backend=backend),
                SummarizationMiddleware(
                    model=model,
                    max_tokens_before_summary=300000,
                    messages_to_keep=6,
                ),
                AnthropicPromptCachingMiddleware(unsupported_model_behavior="ignore"),
                PatchToolCallsMiddleware(),
            ],
            default_interrupt_on=interrupt_on,
            general_purpose_agent=True,
        ),
        SummarizationMiddleware(
            model=model,
            max_tokens_before_summary=300000,
            messages_to_keep=6,
        ),
        AnthropicPromptCachingMiddleware(unsupported_model_behavior="ignore"),
        PatchToolCallsMiddleware(),
    ]

    #agent_middleware.extend(deepagent_middleware)

    if interrupt_on is not None:
        deepagent_middleware.append(HumanInTheLoopMiddleware(interrupt_on=interrupt_on))
        
        #deepagent_middleware.append(ToolMonitoringMiddleware())
        #deepagent_middleware.append(LoggingMiddleware())


    return create_agent(
        model,
        system_prompt=system_prompt,
        tools=tools,
        middleware=deepagent_middleware,
        response_format=response_format,
        context_schema=context_schema,
        checkpointer=checkpointer,
        store=store,
        debug=debug,
        name=name,
        cache=cache,
    ).with_config({"recursion_limit": 1000})

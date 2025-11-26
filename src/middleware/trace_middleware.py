"""
追踪中间件
用于调试和监控 Agent 的执行过程
"""

from src.config.logger import get_logger
logger = get_logger(__name__)
import json
import time
from typing import Callable, Any
from langchain.agents.middleware import wrap_model_call, wrap_tool_call  # type: ignore
from langchain.agents.middleware import ModelRequest, ModelResponse  # type: ignore
from langchain_core.messages import AIMessage  # type: ignore


def _print_message(i, msg):
    """打印单条消息的辅助函数"""
    msg_type = getattr(msg, 'type', 'unknown')
    content = getattr(msg, 'content', str(msg))
    
    # System 消息显示前500字符，User/AI 消息显示更多
    if msg_type == 'system':
        logger.info(f"[{i}] System Prompt:\n{content[:50]}{'...' if len(content) > 50 else ''}\n")
    elif msg_type == 'human':
        logger.info(f"[{i}] User:\n{content[:100]}{'...' if len(content) > 100 else ''}\n")
    elif msg_type == 'ai':
        logger.info(f"[{i}] AI:\n{content[:100] if content else '(tool calls only)'}{'...' if len(content) > 100 else ''}\n")
    elif msg_type == 'tool':
        tool_name = getattr(msg, 'name', 'unknown')
        logger.info(f"[{i}] Tool ({tool_name}):\n{str(content)[:100]}{'...' if len(str(content)) > 100 else ''}\n")


@wrap_model_call
def trace_model_call(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """拦截 LLM 调用，记录输入/输出/耗时"""
    t0 = time.time()
    
    # 从 request 中获取 messages
    messages = getattr(request, "messages", [])
    if not messages:
        # 尝试从 runtime 中获取
        runtime = getattr(request, "runtime", None)
        if runtime and hasattr(runtime, "state"):
            messages = runtime.state.get("messages", [])
    logger.info("[LLM START]")
    
    # 注意：此次是日志显示的逻辑
    if len(messages) <= 2:
        # 消息少，全部打印
        for i, msg in enumerate(messages):
            _print_message(i, msg)
    else:
        # 消息多，只打印首尾
        # 第一条（System Prompt）
        _print_message(0, messages[0])
        
        # 中间省略
        logger.info(f"  ... (skip {len(messages) - 2} messages) ...\n")
        
        # 最后一条（最新消息）
        _print_message(len(messages) - 1, messages[-1])
    
    # 执行真正的模型调用
    resp = handler(request)
    
    dt = (time.time() - t0) * 1000
    logger.info(f"\n [LLM END] {dt:.1f} ms")
    
    # 打印 AI 响应
    ai_message = getattr(resp, "message", resp)
    if isinstance(ai_message, AIMessage):
        content = ai_message.content or ""
        logger.info(f"AIMessage content:\n{content[:100]}{'...' if len(content) > 100 else ''}")
        if getattr(ai_message, "tool_calls", None):
            logger.info(f"\nTool calls requested: {len(ai_message.tool_calls)}")
            for i, tc in enumerate(ai_message.tool_calls, 1):
                logger.info(f"  [{i}] {tc.get('name', 'unknown')}")
                args = tc.get('args', {})
                args_str = json.dumps(args, ensure_ascii=False, indent=2)
                logger.info(f" Args: {args_str[:100]}{'...' if len(args_str) > 100 else ''}")
    else:
        logger.info(f"Raw output:\n{str(ai_message)[:100]}")
    
    return resp


@wrap_tool_call
def trace_tool_call(
    request,
    handler: Callable,
) -> Any:
    """拦截工具调用，记录入参/结果/耗时
    
    对 get_all_tables_info 和 get_table_schema 进行只显示摘要）
    其他工具显示详细输出
    """
    # 从 request 中提取工具名和参数
    tool_call = getattr(request, "tool_call", None)
    if tool_call:
        tool_name = tool_call.get("name", "unknown")
        tool_input = tool_call.get("args", {})
    else:
        tool_name = getattr(request, "tool_name", "unknown")
        tool_input = getattr(request, "tool_input", {})
    
    # 定义需要精简打印的工具
    compact_tools = {"get_all_tables_info", "get_table_schema"}
    is_compact = tool_name in compact_tools

    logger.info(f"[TOOL START] {tool_name}")

    # 打印工具参数
    args_str = json.dumps(tool_input, ensure_ascii=False, indent=2)
    logger.info(f"Args:\n{args_str[:600]}{'...' if len(args_str) > 600 else ''}")
    
    t0 = time.time()
    result = handler(request)
    dt = (time.time() - t0) * 1000
    
    logger.info(f"\n[TOOL END] {dt:.1f} ms")
    
    # 打印工具输出
    if hasattr(result, "content"):
        preview = str(result.content)
    else:
        preview = str(result)
    
    if is_compact:
        # 精简打印：显示摘要
        _print_compact_output(tool_name, preview)
    else:
        # 详细打印：显示完整内容（截断）
        logger.info(f"Output:\n{preview[:100]}{'...' if len(preview) > 100 else ''}")
    
    return result


def _print_compact_output(tool_name: str, text: str) -> None:
    """精简打印工具输出（针对 get_all_tables_info 和 get_table_schema）"""
    import re
    
    logger.info(f"Output (精简模式):")
    
    # 提取关键统计信息
    stats = []
    
    # 统计表数量
    table_count_match = re.search(r"表数量\s*:\s*(\d+)", text)
    if table_count_match:
        stats.append(f"表数量: {table_count_match.group(1)}")
    
    # 统计列数
    col_matches = re.findall(r"列数\s*:\s*(\d+)", text)
    if col_matches:
        total_cols = sum(int(c) for c in col_matches)
        stats.append(f"总列数: {total_cols}")
    
    # 统计 DDL 语句
    ddl_count = text.count("CREATE TABLE")
    if ddl_count:
        stats.append(f"DDL 语句: {ddl_count} 个")
    
    # 提取表名列表
    table_names_match = re.findall(r"^表名\s*:\s*(.+)$", text, flags=re.MULTILINE)
    if table_names_match:
        table_list = ", ".join(table_names_match[:5])
        if len(table_names_match) > 5:
            table_list += f" ... (共{len(table_names_match)}张)"
        stats.append(f"表清单: {table_list}")
    
    # 打印统计信息
    if stats:
        for stat in stats:
            logger.info(f"   {stat}")
    else:
        # 降级：显示前 300 字符
        logger.info(f"   {text[:300]}{'...' if len(text) > 300 else ''}")
    
    # 显示完整字符数
    logger.info(f" 完整输出: {len(text)} 字符")

"""
选择性工具打印器（Selective Tool Printer）
用于精准控制特定工具的打印输出，不影响原始事件流

使用场景：
- get_all_tables_info 和 get_table_schema 输出太长，需要精简打印
- 只打印摘要和关键信息，避免日志冗余
"""

from src.config.logger import get_logger
logger = get_logger(__name__)
import re
from textwrap import shorten
from typing import Set, Optional, Any
from langchain_core.callbacks.base import BaseCallbackHandler


class SelectiveToolPrinter(BaseCallbackHandler):
    """
    选择性工具打印回调
    
    只对指定的工具（如 get_all_tables_info, get_table_schema）进行精简打印
    其他工具和事件保持静默，不影响原有的打印逻辑
    
    特性：
    1. 工具开始时：打印工具名和参数摘要
    2. 工具结束时：
       - 提取表名、DDL数量等关键信息
       - 截断过长内容（按字符数和行数）
       - 显示结构化摘要
    3. 不影响事件流：纯观察者模式，不修改任何消息内容
    """
    
    def __init__(
        self,
        targets: tuple = ("get_all_tables_info", "get_table_schema"),
        max_chars: int = 1200,          # 每段最多打印字符数
        max_lines: int = 40,            # 每段最多打印行数
        show_summary: bool = True,      # 是否显示摘要（表名、DDL数量等）
        show_tool_output: bool = True,  # 是否显示工具输出内容
        echo_non_targets: bool = False, # 是否打印非目标工具
    ):
        """
        初始化选择性打印器
        
        Args:
            targets: 需要精简打印的工具名称集合
            max_chars: 输出内容最大字符数
            max_lines: 输出内容最大行数
            show_summary: 是否显示结构化摘要
            show_tool_output: 是否显示工具输出内容（设为False只显示摘要）
            echo_non_targets: 是否也打印非目标工具
        """
        super().__init__()
        self.targets = set(targets)
        self.max_chars = max_chars
        self.max_lines = max_lines
        self.show_summary = show_summary
        self.show_tool_output = show_tool_output
        self.echo_non_targets = echo_non_targets
        
        # 用于追踪当前工具调用（LangChain callback 体系的限制）
        self._current_tool_name = None
    
    def on_tool_start(
        self, 
        serialized: dict, 
        input_str: str, 
        **kwargs: Any
    ) -> None:
        """工具开始时的回调"""
        name = (serialized or {}).get("name", "unknown")
        self._current_tool_name = name
        
        if name in self.targets or self.echo_non_targets:
            # 精简打印参数
            preview = self._truncate_text(input_str, 200, 5)
            logger.info(f"\n🧰 [工具调用] {name}")
            if preview:
                logger.info(f"   参数: {preview}")
    
    def on_tool_end(
        self, 
        output: str, 
        **kwargs: Any
    ) -> None:
        """工具结束时的回调"""
        # 只处理目标工具
        if self._current_tool_name not in self.targets:
            self._current_tool_name = None
            return
        
        text = str(output) if output is not None else ""
        if not text:
            self._current_tool_name = None
            return
        
        logger.info(f"\n📤 [{self._current_tool_name}] 返回结果:")
        
        # 1. 提取并显示摘要信息
        if self.show_summary:
            self._print_summary(text, self._current_tool_name)
        
        # 2. 显示精简后的内容
        if self.show_tool_output:
            body = self._truncate_text(text, self.max_chars, self.max_lines)
            logger.info("\n" + "─" * 60)
            logger.info(body)
            logger.info("─" * 60)
        
        self._current_tool_name = None
    
    def on_tool_error(
        self, 
        error: Exception, 
        **kwargs: Any
    ) -> None:
        """工具错误时的回调"""
        error_msg = str(error)[:200]
        logger.error(f"\n❌ [工具错误] {self._current_tool_name or 'unknown'}")
        logger.info(f"   错误信息: {error_msg}...")
        self._current_tool_name = None
    
    def _print_summary(self, text: str, tool_name: str) -> None:
        """打印结构化摘要"""
        # 提取表名（从 CREATE TABLE 语句）
        tables = re.findall(
            r"CREATE TABLE\s+([`\"\[]?)([\w\.]+)\1", 
            text, 
            flags=re.IGNORECASE
        )
        table_names = [m[1] for m in tables]
        
        if table_names:
            unique_tables = sorted(set(table_names))
            table_count = len(unique_tables)
            sample_tables = ", ".join(unique_tables[:5])
            if table_count > 5:
                sample_tables += f" ... (共{table_count}张表)"
            
            logger.info(f"   📑 发现表: {sample_tables}")
        
        # 统计结构化段落（常见格式：表名: xxx）
        table_blocks = len(re.findall(r"^表名\s*:", text, flags=re.MULTILINE))
        if table_blocks:
            logger.info(f"   🧾 表清单段落: {table_blocks} 个")
        
        # 统计 DDL 段落
        ddl_blocks = text.count("CREATE TABLE")
        if ddl_blocks:
            logger.info(f"   📐 DDL 语句: {ddl_blocks} 个")
        
        # 统计列数（如果有）
        col_matches = re.findall(r"列数\s*:\s*(\d+)", text)
        if col_matches:
            total_cols = sum(int(c) for c in col_matches)
            logger.info(f"   📊 总列数: {total_cols}")
        
        # 检查是否有文档说明
        if "业务文档说明" in text or "📖" in text:
            doc_count = text.count("业务文档说明") or text.count("[文档")
            logger.info(f"   📖 业务文档: {doc_count} 段")
        
        # 检查是否有历史查询
        if "历史相似查询" in text or "💡" in text:
            sql_count = text.count("[示例") or text.count("SQL:")
            logger.info(f"   💡 历史查询: {sql_count} 个")
    
    def _truncate_text(
        self, 
        text: str, 
        max_chars: int, 
        max_lines: int
    ) -> str:
        """截断文本（按字符数和行数）"""
        if not text:
            return text
        
        # 1. 先按行数截断
        lines = text.splitlines()
        if len(lines) > max_lines:
            lines = lines[:max_lines] + ["... (输出已截断，共{}行)".format(len(text.splitlines()))]
        text2 = "\n".join(lines)
        
        # 2. 再按字符数截断
        if len(text2) > max_chars:
            text2 = shorten(
                text2, 
                width=max_chars, 
                placeholder=" ... (内容过长已截断)"
            )
        
        return text2
    
    # ==================== 其他事件全部静默 ====================
    
    def on_llm_start(self, *args, **kwargs) -> None: pass
    
    def on_llm_end(self, *args, **kwargs) -> None: pass
    
    def on_llm_new_token(self, *args, **kwargs) -> None: pass
    
    def on_chain_start(self, *args, **kwargs) -> None: pass
    
    def on_chain_end(self, *args, **kwargs) -> None: pass
    
    def on_agent_action(self, *args, **kwargs) -> None: pass
    
    def on_agent_finish(self, *args, **kwargs) -> None: pass


# ==================== 便捷工厂函数 ====================

def create_selective_printer(
    mode: str = "minimal",
    targets: tuple = ("get_all_tables_info", "get_table_schema")
) -> SelectiveToolPrinter:
    """
    创建选择性打印器（预设模式）
    
    Args:
        mode: 打印模式
            - "minimal": 最小化（只显示摘要，不显示内容）
            - "compact": 紧凑模式（显示摘要 + 少量内容）
            - "detailed": 详细模式（显示摘要 + 较多内容）
        targets: 需要控制的工具名称
    
    Returns:
        SelectiveToolPrinter 实例
    """
    mode_configs = {
        "minimal": {
            "max_chars": 0,
            "max_lines": 0,
            "show_summary": True,
            "show_tool_output": False,
        },
        "compact": {
            "max_chars": 300,
            "max_lines": 10,
            "show_summary": True,
            "show_tool_output": True,
        },
        "detailed": {
            "max_chars": 1000,
            "max_lines": 30,
            "show_summary": True,
            "show_tool_output": True,
        },
    }
    
    config = mode_configs.get(mode, mode_configs["compact"])
    
    return SelectiveToolPrinter(
        targets=targets,
        **config
    )

from langchain.tools.tool_node import ToolCallRequest
from langchain.agents.middleware import AgentMiddleware
from langchain.messages import ToolMessage
from langgraph.types import Command
from typing import Callable

from src.config.logger import get_logger
logger = get_logger(__name__)


class ToolMonitoringMiddleware(AgentMiddleware):
    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        logger.info(f"Executing tool: {request.tool_call['name']}")
        logger.info(f"Arguments: {request.tool_call['args']}")
        try:
            result = handler(request)
            logger.info(f"Tool completed successfully")
            return result
        except Exception as e:
            logger.error(f"Tool failed: {e}")
            raise

# ============================================================================
# 示例 1：最简单的中间件
"""
  关键规则

  1. 必须继承 AgentMiddleware ← 这个固定
  2. 方法名固定 (before_model, after_model) ← 这个固定
  3. 类名随意 ← 这个不固定

  LangGraph 只看：
  - 是否继承 AgentMiddleware？
  - 是否有 before_model / after_model 方法？
  """
# ============================================================================
class LoggingMiddleware(AgentMiddleware): # ✅ 类名随意
    """
    日志中间件 - 记录每次模型调用

    before_model: 模型调用前执行
    after_model: 模型响应后执行
    """

    def before_model(self, state, runtime):
        """模型调用前"""
        logger.info("\n[中间件] before_model: 准备调用模型")
        logger.info(f"[中间件] 当前消息数: {len(state.get('messages', []))}")
        return None  # 返回 None 表示继续正常流程

    def after_model(self, state, runtime):
        """模型响应后"""
        logger.info("[中间件] after_model: 模型已响应")
        last_message = state.get('messages', [])[-1]
        logger.info(f"[中间件] 响应类型: {last_message.__class__.__name__}")    
        #logger.info(f"[中间件] 消耗 tokens: {runtime.get('usage', {}).get('total_tokens', 0)}")
        return None  # 返回 None 表示不修改状态


# ============================================================================
# 示例 2：修改状态的中间件
# ============================================================================
class CallCounterMiddleware(AgentMiddleware):
    """
    计数中间件 - 统计模型调用次数

    在中间件内部维护计数器（简单版本）
    """

    def __init__(self):
        super().__init__()
        self.count = 0  # 简单计数器

    def after_model(self, state, runtime):
        """模型响应后，增加计数"""
        self.count += 1
        logger.info(f"\n[计数器] 模型调用次数: {self.count}")
        return None  # 不修改 state

# ============================================================================
# 示例 3：消息修剪中间件
# ============================================================================
class MessageTrimmerMiddleware(AgentMiddleware):
    """
    消息修剪中间件 - 限制消息数量

    before_model 修改消息列表
    注意：需要配合无 checkpointer 使用，否则历史会被恢复
    """

    def __init__(self, max_messages=5):
        super().__init__()
        self.max_messages = max_messages
        self.trimmed_count = 0  # 统计修剪次数

    def before_model(self, state, runtime):
        """模型调用前，修剪消息"""
        messages = state.get('messages', [])

        if len(messages) > self.max_messages:
            # 保留最近的 N 条消息
            trimmed_messages = messages[-self.max_messages:]
            self.trimmed_count += 1
            logger.info(f"\n[修剪] 消息从 {len(messages)} 条减少到 {len(trimmed_messages)} 条 (第{self.trimmed_count}次修剪)")
            return {"messages": trimmed_messages}

        return None


# ============================================================================
# 示例 4：输出验证中间件
# ============================================================================
class OutputValidationMiddleware(AgentMiddleware):
    """
    输出验证中间件 - 检查响应长度

    after_model 验证输出
    """

    def __init__(self, max_length=100):
        super().__init__()
        self.max_length = max_length

    def after_model(self, state, runtime):
        """模型响应后，验证输出"""
        messages = state.get('messages', [])
        if not messages:
            return None

        last_message = messages[-1]
        content = getattr(last_message, 'content', '')

        if len(content) > self.max_length:
            logger.warning(f"\n[警告] 响应内容过长 ({len(content)} 字符)，已截断到 {self.max_length}")
            # 这里可以实现截断或重试逻辑

        return None


# ============================================================================
# 示例 5：多个中间件组合
# ============================================================================
class TimingMiddleware(AgentMiddleware):
    """计时中间件"""

    def before_model(self, state, runtime):
        import time
        # 记录开始时间（实际应该用 runtime 的上下文管理）
        logger.info("\n[计时] 开始调用模型...")
        return None

    def after_model(self, state, runtime):
        logger.info("[计时] 模型调用完成")      
        return None



# ============================================================================
# 示例 6：条件跳转（高级）
# ============================================================================
class MaxCallsMiddleware(AgentMiddleware):
    """
    最大调用限制中间件

    通过抛出异常来阻止模型调用（更可靠的方式）
    """

    def __init__(self, max_calls=3):
        super().__init__()
        self.max_calls = max_calls
        self.count = 0  # 简单计数器

    def before_model(self, state, runtime):
        """检查调用次数，超过限制则抛出异常"""
        if self.count >= self.max_calls:
            logger.warning(f"\n[限制] 已达到最大调用次数 {self.max_calls}，停止调用")
            # 抛出自定义异常来阻止继续执行
            raise ValueError(f"已达到最大调用次数限制: {self.max_calls}")

        logger.info(f"[限制] 当前调用次数: {self.count}/{self.max_calls}")
        return None

    def after_model(self, state, runtime):
        """增加计数"""
        self.count += 1
        logger.info("次数+1")
        return None

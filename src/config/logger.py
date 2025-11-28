from loguru import logger
import sys
from pathlib import Path
import json
from datetime import datetime

# 创建基础日志目录
base_log_dir = Path("logs")
base_log_dir.mkdir(exist_ok=True)

# 获取当前日期作为子目录名
current_date = datetime.now().strftime('%Y-%m-%d')
log_dir = base_log_dir / current_date
log_dir.mkdir(exist_ok=True)

# 移除默认的控制台输出
logger.remove()

# 添加控制台输出
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> |<cyan>{file}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO"
)

# 添加文件输出 - 直接使用固定路径（兼容旧版本loguru）
logger.add(
    str(log_dir / f"app_{current_date}.log"),
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {file}:{function}:{line} - {message}",
    level="INFO"
)

# 错误日志单独存储
logger.add(
    str(log_dir / f"error_{current_date}.log"),
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {file}:{function}:{line} - {message}",
    level="ERROR"
)

def get_logger(service: str):
    """获取带有服务名称的 logger"""
    return logger.bind(service=service)

def log_structured(event_type: str, data: dict):
    """结构化日志记录"""
    logger.info({"event_type": event_type, "data": data}) 
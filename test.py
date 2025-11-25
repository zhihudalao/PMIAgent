from src.config.logger import get_logger
from src.llms.llm import get_llm_by_type

def main():
    logger = get_logger(__name__)
    llm = get_llm_by_type("basic")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "你是谁？"},
    ]
    try:
        # 使用流式调用，因为 enable_thinking 参数只支持流式调用
        # response = ""
        # for chunk in llm.stream(messages):
        #     if hasattr(chunk, 'content'):
        #         response += chunk.content
        #         print(chunk.content, end="", flush=True)

        # logger.info(f"response: {response}")
        response = llm.invoke(messages)
        full_response = response.model_dump_json(indent=4, exclude_none=True)
        logger.info(f"full_response: {full_response}")
        print("\n")  # 添加换行

    except Exception as e:
        logger.error(f"error: {e}")
        print(e)


if __name__ == "__main__":
    main()

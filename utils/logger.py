import logging
import sys

def get_logger(name: str = "delivery_ai") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # 이미 설정된 로거 재사용

    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    fmt = "[%(asctime)s] [%(levelname)s] %(name)s - %(message)s"
    handler.setFormatter(logging.Formatter(fmt))

    logger.addHandler(handler)
    return logger

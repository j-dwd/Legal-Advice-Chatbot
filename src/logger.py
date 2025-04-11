from loguru import logger

def setup_logger():
    logger.add("app.log", format="{time} {level} {message}", level="INFO")
    return logger
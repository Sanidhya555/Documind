import logging, sys, os

def setup_logger(name: str = "rag_app") -> logging.Logger:
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger
    
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    
    logger.setLevel(level)
    
    h = logging.StreamHandler(sys.stdout)
    
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    
    logger.addHandler(h)
    logger.propagate = False
    
    return logger

import logging
import colorlog
from typing import Optional

def setup_logging(name: str, level: Optional[int] = logging.INFO) -> logging.Logger:
    """
    Set up logging configuration with colorlog
    
    Args:
        name (str): Logger name to be used for the logging instance
        level (Optional[int]): Logging level to be set. Defaults to logging.INFO
        
    Returns:
        logging.Logger: Configured logger instance with color formatting
    
    Example:
        logger = setup_logging("my_module")
        logger.info("This will be displayed in green")
        logger.error("This will be displayed in red")
    """
    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter(
        '%(log_color)s%(asctime)s - %(levelname)s - %(message)s%(reset)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        log_colors={
            'DEBUG':    'cyan',
            'INFO':     'green',
            'WARNING':  'yellow',
            'ERROR':    'red',
            'CRITICAL': 'red,bg_white',
        }
    ))

    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove any existing handlers to avoid duplicate logs
    for existing_handler in logger.handlers[:]:
        logger.removeHandler(existing_handler)
    
    logger.addHandler(handler)
    return logger
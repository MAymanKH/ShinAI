import logging
import sys
import os

def setup_logger(name="ShinAI", log_file="shinai_bot.log", level=logging.INFO):
    """
    Sets up a logger with both file and console handlers
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Check if handlers already exist to avoid duplicate logs
    if not logger.handlers:
        # Create handlers
        c_handler = logging.StreamHandler(sys.stdout)
        f_handler = logging.FileHandler(log_file, encoding='utf-8')
        
        c_handler.setLevel(level)
        f_handler.setLevel(level)
        
        # Create formatters and add it to handlers
        # Using a format that includes timestamp, name, level, and message
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        
        c_handler.setFormatter(formatter)
        f_handler.setFormatter(formatter)
        
        # Add handlers to the logger
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)

    return logger

# Create a global logger instance
logger = setup_logger()

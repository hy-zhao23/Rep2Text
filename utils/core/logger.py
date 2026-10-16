import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
import os
from typing import Optional

class CustomLogger:
    _instance: Optional[logging.Logger] = None
    
    @classmethod
    def setup_logger(
        cls,
        name: str = 'project_logger',
        console_level: int = logging.WARNING,
        file_level: int = logging.DEBUG
    ) -> logging.Logger:
        """
        Configure and return a logger instance with both console and file handlers
        """
        if cls._instance is not None:
            return cls._instance
            
        # Create logger instance
        logger = logging.getLogger(name)
        logger.setLevel(min(console_level, file_level))  # Set to lowest level among handlers
        
        # Prevent adding handlers multiple times
        if logger.handlers:
            return logger
            
        # Create formatters
        console_formatter = logging.Formatter(
            fmt='[%(asctime)s] %(levelname)-5s %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        file_formatter = logging.Formatter(
            fmt='[%(asctime)s] %(levelname)-5s [%(filename)s:%(lineno)d] %(message)s',
            datefmt='%m-%d %H:%M:%S'
        )
        
        # Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level)
        console_handler.setFormatter(console_formatter)
        
        # File Handler
        try:
            log_dir = Path(os.getenv('PROJ')) / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file: str = log_dir / f"{os.getenv('SLURM_JOB_NAME')}_{os.getenv('SLURM_JOB_ID')}.log"

            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=100 * 1024 * 1024,  # 100 MB
                backupCount=5,
                encoding='utf-8'
            )
            file_handler.setLevel(file_level)
            file_handler.setFormatter(file_formatter)
            
            # Add handlers to logger
            logger.addHandler(console_handler)
            logger.addHandler(file_handler)
            
        except Exception as e:
            logger.error(f"Failed to create file handler: {str(e)}")
            # If file handler fails, still keep console handler
            logger.addHandler(console_handler)
        
        cls._instance = logger
        return logger

# Create default logger instance
logger = CustomLogger.setup_logger()
#!/usr/bin/env python3
"""
Centralized logging module for the agentRAG project.

This module provides a standardized logging class with pretty formatting
that can be imported and used across all Python files in the project.

Usage Examples:
    # Basic usage in any Python file:
    from logger import AgentLogger
    logger = AgentLogger(__name__)
    logger.info('Your message here')
    
    # Different log levels:
    logger.debug('Debug info')
    logger.info('General info') 
    logger.warning('Warning message')
    logger.error('Error message')
    logger.critical('Critical error')
    
    # Change log level:
    logger.set_level('DEBUG')  # Show debug messages
    
    # Backward compatibility:
    from logger import doLogging
    doLogging('info', 'Your message')

Features:
    - Color-coded log levels with emojis for visual distinction
    - Consistent timestamp formatting across all modules
    - Terminal-friendly output with ANSI colors
    - Integration with Python's standard logging system
    - Easy import and standardization across the project
    - Configurable log levels per logger instance
"""

import datetime
import logging
import os
from typing import Optional


class AgentLogger:
    """
    Centralized logging class with pretty formatting for terminal output.
    
    Features:
    - Color-coded log levels with emojis
    - Consistent timestamp formatting
    - Terminal-friendly output
    - Integration with Python's logging system
    - Configurable log levels
    """
    
    def __init__(self, name: str = __name__, level: str = "INFO", enable_python_logging: bool = False):
        """
        Initialize the AgentLogger.
        
        Args:
            name: Logger name (typically __name__ of the calling module)
            level: Default log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            enable_python_logging: Whether to also log to Python's logging system (default: False)
        """
        self.name = name
        self.level = level.upper()
        self.enable_python_logging = enable_python_logging
        
        # Color codes for different log levels
        self.colors = {
            'DEBUG': '\033[36m',    # Cyan
            'INFO': '\033[32m',     # Green
            'WARNING': '\033[33m',  # Yellow
            'ERROR': '\033[31m',    # Red
            'CRITICAL': '\033[35m', # Magenta
        }
        
        # Symbols for different log levels
        self.symbols = {
            'DEBUG': '🔍',
            'INFO': 'ℹ️ ',
            'WARNING': '⚠️ ',
            'ERROR': '❌',
            'CRITICAL': '🚨',
        }
        
        # Reset color code
        self.reset_color = '\033[0m'
        
        # Set up Python logging compatibility (only if enabled)
        if self.enable_python_logging:
            self.python_logger = logging.getLogger(name)
            self._setup_python_logger()
        else:
            self.python_logger = None
    
    def _setup_python_logger(self):
        """Set up Python's logging system for compatibility."""
        if not self.python_logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.python_logger.addHandler(handler)
            self.python_logger.setLevel(getattr(logging, self.level, logging.INFO))
    
    def _format_message(self, level: str, message: str) -> str:
        """
        Format a log message with colors, symbols, and timestamp.
        
        Args:
            level: Log level
            message: Message to format
            
        Returns:
            Formatted message string
        """
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        level_upper = level.upper()
        color = self.colors.get(level_upper, '')
        symbol = self.symbols.get(level_upper, '')
        
        # Create pretty formatted message
        formatted_message = f"{color}[{timestamp}] {symbol} {level_upper:8} | {message}{self.reset_color}"
        return formatted_message
    
    def _log(self, level: str, message: str):
        """
        Internal logging method.
        
        Args:
            level: Log level
            message: Message to log
        """
        # Print pretty formatted message to terminal
        formatted_message = self._format_message(level, message)
        print(formatted_message)
        
        # Also log to Python's logging system if enabled
        if self.enable_python_logging and self.python_logger:
            log_method = getattr(self.python_logger, level.lower(), self.python_logger.info)
            log_method(message)
    
    def debug(self, message: str):
        """Log a debug message."""
        self._log('DEBUG', message)
    
    def info(self, message: str):
        """Log an info message."""
        self._log('INFO', message)
    
    def warning(self, message: str):
        """Log a warning message."""
        self._log('WARNING', message)
    
    def error(self, message: str):
        """Log an error message."""
        self._log('ERROR', message)
    
    def critical(self, message: str):
        """Log a critical message."""
        self._log('CRITICAL', message)
    
    def set_level(self, level: str):
        """
        Set the logging level.
        
        Args:
            level: New log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        self.level = level.upper()
        if self.python_logger:
            self.python_logger.setLevel(getattr(logging, self.level, logging.INFO))
    
    def get_level(self) -> str:
        """Get the current logging level."""
        return self.level


# Convenience function for backward compatibility
def create_logger(name: str = __name__, level: str = "INFO", enable_python_logging: bool = False) -> AgentLogger:
    """
    Create and return an AgentLogger instance.
    
    Args:
        name: Logger name (typically __name__ of the calling module)
        level: Default log level
        enable_python_logging: Whether to also log to Python's logging system
        
    Returns:
        AgentLogger instance
    """
    return AgentLogger(name, level, enable_python_logging)


# Default logger instance for simple usage
default_logger = AgentLogger("agentRAG")


def doLogging(level: str, message: str):
    """
    Backward compatibility function for the old doLogging interface.
    
    Args:
        level: Log level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        message: The message to log
    """
    default_logger._log(level, message)


if __name__ == "__main__":
    # Test the logging functionality
    logger = AgentLogger("test_logger")
    
    print("Testing AgentLogger with different log levels:\n")
    
    logger.debug("This is a debug message for troubleshooting")
    logger.info("Application started successfully")
    logger.info("✓ Processing completed - 150 documents loaded")
    logger.warning("⚠ Configuration file not found, using defaults")
    logger.error("❌ Failed to connect to database")
    logger.critical("🚨 System critical error - shutting down")
    
    print("\nTesting backward compatibility function:\n")
    doLogging("info", "This uses the old doLogging interface")
    doLogging("warning", "But now it's powered by the AgentLogger class")
    
    print("\nFeatures:")
    print("- Color-coded log levels with emojis")
    print("- Consistent timestamp formatting")
    print("- Class-based design for easy import and reuse")
    print("- Backward compatibility with old doLogging function")
    print("- Integration with Python's logging system")
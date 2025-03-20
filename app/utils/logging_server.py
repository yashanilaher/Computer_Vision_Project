# logging_server.py

import argparse
from pathlib import Path
import zmq
from loguru import logger
from config_types import LoggingConfigs
import os


def set_logging_configs(logging_configs: LoggingConfigs) -> None:
    """
    Configure the logging server with the provided settings.

    Args:
        logging_configs: Configuration for logging (e.g., file name, rotation, etc.).
    """
    # Remove the default logger configuration
    logger.remove()

    # Get the absolute path to the log file in utils/logs/
    log_file_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), logging_configs.log_file_name)
    )
    print(f"Logging to: {log_file_path}")

    # Add a file logger with rotation and compression
    logger.add(
        log_file_path,
        rotation=logging_configs.log_rotation,  # Rotate logs at midnight
        compression=logging_configs.log_compression,  # Compress rotated logs
        format=logging_configs.server_log_format,
        level=logging_configs.min_log_level,
        enqueue=True,  # Ensure thread-safe logging
        backtrace=True,  # Enable detailed error traces
        diagnose=True,  # Enable exception diagnostics
    )



def start_logging_server(logging_configs: LoggingConfigs) -> None:
    """
    Start the logging server to receive log messages from clients.

    Args:
        logging_configs: Configuration for logging (e.g., server port).
    """
    # Create a ZeroMQ socket for subscribing to log messages
    socket = zmq.Context().socket(zmq.SUB)
    socket.bind(f"tcp://127.0.0.1:{logging_configs.log_server_port}")
    socket.subscribe("")  # Subscribe to all messages

    logger.info("Logging server started and listening for messages...")

    while True:
        try:
            # Receive log messages from clients
            ret_val = socket.recv_multipart()
            log_level_name, message = ret_val

            # Decode the log level and message
            log_level_name = log_level_name.decode("utf-8").strip()
            message = message.decode("utf-8").strip()

            # Log the message using the configured logger
            logger.log(log_level_name, message)

        except Exception as ex:
            logger.error(f"Error processing log message: {ex}")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Start the logging server.")
    parser.add_argument("--config_file_path", required=True, help="Path to the logging configuration file.")
    args = parser.parse_args()

    # Load logging configurations from the TOML file
    config_file_path = Path(args.config_file_path)
    logging_configs = LoggingConfigs.load_from_path(config_file_path)

    # Set up the logging configurations and start the server
    set_logging_configs(logging_configs)
    start_logging_server(logging_configs)
# logging_client.py

from __future__ import annotations  # Needed for type hints

from typing import TYPE_CHECKING

import zmq
from zmq.log.handlers import PUBHandler

if TYPE_CHECKING:
    from config_types import LoggingConfigs
    from loguru import Logger


def setup_network_logger_client(logging_configs: LoggingConfigs, logger: Logger) -> None:
    """Set up a network logger client that sends log messages to a logging server.

    Args:
        logging_configs: Configuration for logging (e.g., server port).
        logger: The logger instance to configure.

    """
    # Create a ZeroMQ socket for publishing logs
    zmq_socket = zmq.Context().socket(zmq.PUB)
    zmq_socket.connect(f"tcp://127.0.0.1:{logging_configs.log_server_port}")

    # Add a PUBHandler to the logger
    handler = PUBHandler(zmq_socket)

    # Remove the default logger configuration (to avoid duplicate logs)
    logger.remove()

    # Add the network handler with the specified format and log level
    logger.add(
        handler,
        format=logging_configs.client_log_format,
        enqueue=True,  # Ensure thread-safe logging
        level=logging_configs.min_log_level,
        backtrace=True,  # Enable detailed error traces
        diagnose=True,  # Enable exception diagnostics
    )

# config_types.py

from pathlib import Path
from typing import Literal
import tomllib
from pydantic import BaseModel, ConfigDict


def load_toml(file_name: Path) -> dict:
    """Load a TOML file and return its contents as a dictionary."""
    with file_name.open("rb") as file_obj:
        return tomllib.load(file_obj)


class LoggingConfigs(BaseModel):
    """Configuration model for logging settings."""

    model_config = ConfigDict(extra="forbid")  # Prevent extra fields

    # Logging settings
    min_log_level: Literal[
        "TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"
    ] = "DEBUG"
    log_server_port: int = 9999
    server_log_format: str = "[{level}] | {message}"
    client_log_format: str = "{time:YYYY-MM-DD HH:mm:ss} | {file}: {line} | {message}"
    log_rotation: str = "00:00"  # Rotate logs at midnight
    log_file_name: str = "logs/logs.txt"
    log_compression: str = "zip"  # Compress rotated logs

    @staticmethod
    def load_from_path(file_path: str) -> "LoggingConfigs":
        """Load logging configurations from a TOML file."""
        configs: LoggingConfigs = LoggingConfigs.model_validate(load_toml(Path(file_path)))
        return configs
"""Project configuration"""
from pathlib import Path

from dynaconf import Dynaconf, add_converter

add_converter("path", Path)

settings = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=["settings.toml", ".secrets.toml"],
)

"""
config.py
=========
Tiny helper to load the YAML config. Reads with explicit UTF-8 encoding so it
works identically on Windows (which otherwise defaults to cp1252 and crashes on
any non-ASCII byte) and Linux/macOS.
"""

import yaml


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

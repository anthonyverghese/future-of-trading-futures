"""
clear_cache.py — Deletes the local session cache.

Run with: python clear_cache.py
"""

import os
from cache import CACHE_PATH

if os.path.exists(CACHE_PATH):
    os.remove(CACHE_PATH)
    print("Session cache cleared.")
else:
    print("No cache found.")

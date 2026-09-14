#!/usr/bin/env python3
"""
Alias to demo.py from scripts directory.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import demo

if __name__ == "__main__":
    demo.main()

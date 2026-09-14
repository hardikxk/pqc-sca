# Default: Fast smoke demo + launches web dashboard
uv run demo.py

# Run full-size training and benchmarking
uv run demo.py --full

# Terminal-only mode (runs the full pipeline and exits without web server)
uv run demo.py --cli-only

# Skip pipeline rerun and immediately start dashboard from existing artifacts
uv run demo.py --skip-pipeline

# Run on a custom port without opening browser
uv run demo.py --port 8080 --no-browser

#!/usr/bin/env python3
"""
Entry point for running the Big 2 web server.
"""

import argparse
import sys

import uvicorn

from big2.web_api import app


def main():
    """Main entry point for the web server."""
    parser = argparse.ArgumentParser(description="Run the Big 2 web interface")
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("🎴 Big 2 Web Interface")
    print("=" * 80)
    print(f"\nStarting server on http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop\n")

    try:
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            reload=args.reload,
        )
    except KeyboardInterrupt:
        print("\n\nServer stopped. Thanks for playing!")
        sys.exit(0)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Export OpenAPI schema from the FastAPI application."""

import json
import sys
from pathlib import Path

from big2.game_server.main import app


def export_openapi(output_file: str = "openapi.json", format: str = "json") -> None:
    """
    Export the OpenAPI schema to a file.

    Args:
        output_file: Output file path
        format: Output format (json or yaml)
    """
    schema = app.openapi()

    output_path = Path(output_file)

    if format == "json":
        with output_path.open("w") as f:
            json.dump(schema, f, indent=2)
        print(f"✓ OpenAPI schema exported to {output_path} (JSON)")
    elif format == "yaml":
        try:
            import yaml

            with output_path.open("w") as f:
                yaml.dump(schema, f, default_flow_style=False, sort_keys=False)
            print(f"✓ OpenAPI schema exported to {output_path} (YAML)")
        except ImportError:
            print("ERROR: PyYAML not installed. Install with: uv pip install pyyaml")
            sys.exit(1)
    else:
        print(f"ERROR: Unsupported format '{format}'. Use 'json' or 'yaml'.")
        sys.exit(1)

    # Print summary
    print("\nOpenAPI Info:")
    print(f"  Title: {schema['info']['title']}")
    print(f"  Version: {schema['info']['version']}")
    print(f"  Endpoints: {len(schema['paths'])}")
    print("\nEndpoints:")
    for path, methods in schema["paths"].items():
        for method in methods.keys():
            if method != "parameters":
                print(f"  {method.upper():6} {path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export OpenAPI schema")
    parser.add_argument(
        "-o",
        "--output",
        default="openapi.json",
        help="Output file path (default: openapi.json)",
    )
    parser.add_argument(
        "-f",
        "--format",
        choices=["json", "yaml"],
        default="json",
        help="Output format (default: json)",
    )

    args = parser.parse_args()
    export_openapi(args.output, args.format)

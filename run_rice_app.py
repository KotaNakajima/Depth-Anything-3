"""
Convenience launcher for the Rice Canopy GUI without installing the package.

Usage:
  # In project root (where this file and the 'src' directory exist)
  python run_rice_app.py
"""
import os
import sys

# Ensure 'src' is on sys.path so 'depth_anything_3' can be imported in src-layout projects
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from depth_anything_3.app.rice_canopy_app import main  # noqa: E402


if __name__ == "__main__":
    # You may set host/port via environment variables if needed:
    #   export DA3_RICE_APP_HOST=0.0.0.0
    #   export DA3_RICE_APP_PORT=7861
    main()

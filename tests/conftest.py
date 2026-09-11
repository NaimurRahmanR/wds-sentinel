import sys
from pathlib import Path

# Ensure src/ is importable without requiring an editable install, so tests
# run the same way locally and in CI.
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

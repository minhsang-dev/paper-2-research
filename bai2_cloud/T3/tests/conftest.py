import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import threebin  # noqa: E402,F401  (adds protocol_code/replay to sys.path)

"""Generic evaluation entry point; legacy module retained for compatibility."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.eval_aime26 import main

if __name__ == "__main__":
    main()

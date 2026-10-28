import importlib
import os
import socket
import sys
from pathlib import Path

from dotenv import load_dotenv

ok = True

def report(passed, label, hint=""):
    global ok
    print(("PASS " if passed else "FAIL ") + label)
    if not passed:
        ok = False
        if hint:
            print("     -> " + hint)

v = sys.version_info
report((3,10) <= (v.major, v.minor) < (3,14), f"Python {v.major}.{v.minor}.{v.micro}",
       "Use Python 3.12 or 3.13 (see step 1).")

report(sys.prefix != sys.base_prefix, "Virtual environment is active",
       "Run: source .venv/bin/activate")

for name in ["fastapi", "uvicorn", "google.genai", "chromadb", "dotenv", "multipart", "pypdf"]:
    try:
        importlib.import_module(name)
        report(True, f"Import {name}")
    except Exception as e:
        report(False, f"Import {name}", f"{type(e).__name__}: {e}. Run: pip install -r requirements.txt")

load_dotenv()
key = os.getenv("GEMINI_API_KEY")
report(bool(key), "GEMINI_API_KEY found in .env", "Check that .env exists in this folder (step 6).")
report(bool(os.getenv("GENERATION_MODEL")), "GENERATION_MODEL set in .env", "Add it (step 8a).")
report(bool(os.getenv("EMBEDDING_MODEL")), "EMBEDDING_MODEL set in .env", "Add it (step 8a).")

gi = Path(".gitignore")
report(gi.exists() and ".env" in gi.read_text().splitlines(), ".gitignore lists .env",
       "Add a line containing exactly .env to .gitignore (step 6a).")

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    in_use = s.connect_ex(("127.0.0.1", 8000)) == 0
report(not in_use, "Port 8000 is free", "Another server is using it, Run: lsof -i :8000")

print("\nEverything looks good!" if ok else "\nFix the FAIL lines above, then run this again.")
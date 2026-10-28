import json
import sys
import time

import httpx

URL = "http://127.0.0.1:8000/ask/stream"
question = " ".join(sys.argv[1:]) or "How long do I have to return unopened coffee, and how much is express shipping?"

print(f"Question: {question}\n")
start = time.time()
first_token_at = None
event_name = None

try:
    with httpx.stream("POST", URL, json={"question": question}, timeout=120) as response:
        if response.status_code != 200:
            response.read()
            raise SystemExit(f"HTTP {response.status_code}: {response.text}")

        for line in response.iter_lines():
            if line.startswith("event: "):
                event_name = line[len("event: "):]
            elif line.startswith("data: "):
                data = json.loads(line[len("data: "):])

                if event_name == "retrieved":
                    print(f"[retrieved {len(data['chunks'])} chunks after {time.time() - start:.1f}s]\n")
                elif event_name == "token":
                    if first_token_at is None:
                        first_token_at = time.time() - start
                    print(data["text"], end="", flush=True)  # print without a newline, right now
                elif event_name == "done":
                    print("\n\n--- citations ---")
                    for c in data["citations"]:
                        print(f"[{c['number']}] {c['label']}")
                    if not data["citations"]:
                        print("(none)")
                elif event_name == "error":
                    print(f"\n\nERROR: {data['detail']}")
except httpx.ConnectError:
    raise SystemExit("Could not connect. Is the server running? (uvicorn app.main:app --reload)")

total = time.time() - start
if first_token_at is not None:
    print(f"\nFirst text after {first_token_at:.1f}s, finished after {total:.1f}s")
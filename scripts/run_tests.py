import json
import sys

import httpx

BASE = "http://127.0.0.1:8000"
TEST_FILE = "_selftest_observatory.txt"
TEST_TEXT = (
    "Marlowe Observatory Visitor Notes\n\n"
    "The Marlowe Observatory opens its doors at 6:45 pm on clear Fridays. "
    "Adult admission costs 7 euros, and children under 12 enter free. "
    "Telescope tours last 40 minutes and start every hour."
)

passed = failed = warned = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global passed, failed
    if ok:
        passed += 1
        print(f"PASS  {name}")
    else:
        failed += 1
        print(f"FAIL  {name}  {detail}")


def soft_check(name: str, ok: bool, detail: str = "") -> None:
    """For checks that depend on the AI's wording. A miss is a warning, not a failure."""
    global passed, warned
    if ok:
        passed += 1
        print(f"PASS  {name}")
    else:
        warned += 1
        print(f"WARN  {name}  {detail}")


def main() -> None:
    client = httpx.Client(base_url=BASE, timeout=120)

    # 1. Health
    r = client.get("/health")
    check("GET /health returns 200", r.status_code == 200, r.text)
    check("Gemini key is loaded", r.json().get("gemini_key_loaded") is True, "see Step 6")

    # 2. Upload validation (these never call Gemini)
    r = client.post("/upload", files={"file": ("notes.docx", b"hello", "application/octet-stream")})
    check("Upload rejects .docx with 400", r.status_code == 400, f"got {r.status_code}")

    r = client.post("/upload", files={"file": ("empty.txt", b"", "text/plain")})
    check("Upload rejects empty file with 400", r.status_code == 400, f"got {r.status_code}")

    r = client.post("/upload")
    check("Upload with no file returns 422", r.status_code == 422, f"got {r.status_code}")

    # 3. Real upload
    r = client.post("/upload", files={"file": (TEST_FILE, TEST_TEXT.encode(), "text/plain")})
    check("Upload .txt returns 200", r.status_code == 200, r.text)
    if r.status_code != 200:
        print("\nCannot continue without a successful upload. Fix the error above first.")
        return
    check("Chunks were stored", r.json().get("chunks_stored", 0) >= 1, r.text)

    try:
        # 4. Listing
        r = client.get("/documents")
        names = [d["filename"] for d in r.json().get("documents", [])]
        check("GET /documents lists the new file", TEST_FILE in names, str(names))

        # 5. Ask: answer and citations
        r = client.post("/ask", json={"question": "What time does the Marlowe Observatory open on Fridays?"})
        check("POST /ask returns 200", r.status_code == 200, r.text)
        if r.status_code == 200:
            body = r.json()
            check("Answer contains the right fact (6:45)", "6:45" in body["answer"], body["answer"])
            check("Citations are present", len(body["citations"]) >= 1, "no [n] markers in the answer")
            sources = {c["source"] for c in body["citations"]}
            check("Citation points to the test file", TEST_FILE in sources, str(sources))

        # 6. Ask: validation
        r = client.post("/ask", json={"question": ""})
        check("Empty question returns 422", r.status_code == 422, f"got {r.status_code}")
        r = client.post("/ask", json={"question": "   "})
        check("Blank question returns 400", r.status_code == 400, f"got {r.status_code}")

        # 7. Ask: unrelated question should be refused (depends on the model, so soft)
        r = client.post("/ask", json={"question": "Who won the 2018 FIFA World Cup?"})
        soft_check(
            "Unrelated question is refused",
            r.status_code == 200 and "couldn't find" in r.json().get("answer", "").lower(),
            r.text[:200],
        )

        # 8. Streaming
        events = []
        with client.stream(
            "POST", "/ask/stream", json={"question": "How much does adult admission cost at the Marlowe Observatory?"}
        ) as resp:
            status = resp.status_code
            name = None
            for line in resp.iter_lines():
                if line.startswith("event: "):
                    name = line[len("event: "):]
                elif line.startswith("data: "):
                    events.append((name, json.loads(line[len("data: "):])))
        kinds = [e[0] for e in events]
        check("Stream returns 200", status == 200, f"got {status}")
        check("Stream has retrieved, token and done events",
              "retrieved" in kinds and "token" in kinds and kinds[-1] == "done", str(kinds))
        if kinds and kinds[-1] == "done":
            check("Streamed answer contains the right fact (7)", "7" in events[-1][1]["answer"], events[-1][1]["answer"])

        # 9. Cleanup endpoint guard
        r = client.delete("/documents")
        check("DELETE /documents without confirm returns 400", r.status_code == 400, f"got {r.status_code}")

    finally:
        # 10. Delete the test file (runs even if a check above crashed)
        r = client.delete(f"/documents/{TEST_FILE}")
        check("DELETE /documents/{filename} returns 200", r.status_code == 200, r.text)
        r = client.delete(f"/documents/{TEST_FILE}")
        check("Deleting it again returns 404", r.status_code == 404, f"got {r.status_code}")

    print(f"\n{passed} passed, {failed} failed, {warned} warnings")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except httpx.ConnectError:
        raise SystemExit("Could not connect. Start the server first: uvicorn app.main:app --reload")
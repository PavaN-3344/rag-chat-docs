import os

from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise SystemExit("GEMINI_API_KEY not found. Check your .env file.")

generation_model = os.getenv("GENERATION_MODEL", "gemini-3.6-flash")
embedding_model = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")

client = genai.Client(api_key=api_key)

print(f"Test 1: generation with {generation_model}")
response = client.models.generate_content(
    model = generation_model,
    contents= "In one short sentence, what is a vector database?"
)
print("Answer : ",response.text)

print(f"\nTest 2 : embedding with{embedding_model}")
result = client.models.embed_content(
    model = embedding_model,
    contents= "Hello, this is a test sentence.",
)
vector= result.embeddings[0].values
print("Embedding length: ",len(vector))
print("First 5 numbers:", [round(x,4) for x in vector[:5]])

print("\nAll Gemini tests passed")
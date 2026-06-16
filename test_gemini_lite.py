import os

from google import genai


PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "bloom-475216")
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")
MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-lite")


client = genai.Client(
    vertexai=True,
    project=PROJECT_ID,
    location=LOCATION,
)

response = client.models.generate_content(
    model=MODEL,
    contents="Explain multi-agent human collaboration in one short paragraph.",
)

print(response.text)
client.close()

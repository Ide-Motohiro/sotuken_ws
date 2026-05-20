from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(
    vertexai=True,
    project="gen-lang-client-0698198570",
    location="us-central1",
)

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="こんにちは。一言だけ返して。",
)
print(response.text)

from geminiplayground.core import GeminiClient

from rich import print
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

if __name__ == "__main__":
    model = "models/gemini-1.5-pro-latest"
    gemini_client = GeminiClient()
    prompt = "Write a poem about the ocean"
    response = gemini_client.generate_response(model=model, prompt=prompt)
    print("Gemini: ", response.text)

    response = gemini_client.generate_response(
        model=model, prompt=prompt, timeout=0.1, stream=True
    )
    for candidate in response:
        print("Gemini: ", candidate.text)

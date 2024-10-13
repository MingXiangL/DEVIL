from rich import print

from geminiplayground import GeminiClient
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

if __name__ == "__main__":
    gemini_client = GeminiClient()
    models = gemini_client.query_models()
    for m in models:
        print(m)

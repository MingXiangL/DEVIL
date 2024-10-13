from rich import print
from geminiplayground import GeminiClient

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

if __name__ == "__main__":
    gemini_client = GeminiClient()
    files = gemini_client.query_files(page_size=5)
    print(files)

from rich import print

from geminiplayground import GeminiClient
from dotenv import load_dotenv, find_dotenv
import arrow

load_dotenv(find_dotenv())
if __name__ == "__main__":
    gemini_client = GeminiClient()
    files = gemini_client.query_files()

    for file in files:
        time_to_expire = arrow.now().fromdate(file.expiration_time).humanize()
        print(file.name, file.display_name, time_to_expire, file.mime_type, file.uri)

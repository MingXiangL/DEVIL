from rich import print

from geminiplayground.core import GeminiClient
from geminiplayground.parts import VideoFile, GitRepo
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

if __name__ == "__main__":
    gemini_client = GeminiClient()
    video_path = "../data/transformers-explained.mp4"
    video = VideoFile(video_path, gemini_client=gemini_client)
    repo_url = "https://github.com/karpathy/ng-video-lecture"
    codebase = GitRepo.from_url(
        repo_url,
        branch="master",
        config={
            "content": "code-files",  # "code-files" or "issues"
            "file_extensions": [".py"],
        },
    )
    prompt = [
        "Create a blog post" "Title: Introduction to transformers",
        "based on the following video:",
        video,
        "Also, generate some code snippets from the following codebase, "
        "and include them in the blog post.",
        codebase,
    ]

    model = "models/gemini-1.5-pro-latest"
    # token_count = gemini_client.count_tokens(model, prompt)
    # print("Tokens count: ", token_count)
    response = gemini_client.generate_response(model, prompt, stream=True)

    # Print the response
    for message_chunk in response:
        if message_chunk.parts:
            print(message_chunk.text)

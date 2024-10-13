from rich import print

from geminiplayground.core import GeminiClient
from geminiplayground.parts import VideoFile
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())


def chat_wit_your_video():
    """
    Get the content parts of a video and generate a request.
    :return:
    """
    gemini_client = GeminiClient()
    model_name = "models/gemini-1.5-pro-latest"

    video_file_path = "./../data/transformers-explained.mp4"
    video_file = VideoFile(video_file_path, gemini_client=gemini_client)
    keyframes = video_file.extract_keyframes()
    print(keyframes)

    prompt = [
        "Describe the content of the video",
        video_file,
        "what is the video about?",
    ]
    tokens_count = gemini_client.count_tokens(model_name, prompt)
    print("Tokens count: ", tokens_count)
    response = gemini_client.generate_response(model_name, prompt, stream=True)
    for message_chunk in response:
        if message_chunk.parts:
            print(message_chunk.text)


if __name__ == "__main__":
    chat_wit_your_video()

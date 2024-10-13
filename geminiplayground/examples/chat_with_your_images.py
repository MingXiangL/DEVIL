from rich import print

from geminiplayground.core import GeminiClient
from geminiplayground.parts import ImageFile
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())


def chat_wit_your_images():
    """
    Get the content parts of an image and generate a request.
    :return:
    """
    gemini_client = GeminiClient()

    image_file_path = "https://upload.wikimedia.org/wikipedia/commons/4/47/PNG_transparency_demonstration_1.png"
    image_file = ImageFile(image_file_path, gemini_client=gemini_client)
    prompt = ["what do you see in this image?", image_file]
    model_name = "models/gemini-1.5-pro-latest"
    tokens_count = gemini_client.count_tokens(model_name, prompt)
    print(f"Tokens count: {tokens_count}")
    response = gemini_client.generate_response(model_name, prompt, stream=True)
    for message_chunk in response:
        if message_chunk.parts:
            print(message_chunk.text)


if __name__ == "__main__":
    chat_wit_your_images()

from geminiplayground.core import GeminiClient
from geminiplayground.parts import AudioFile
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())


def chat_wit_your_audios():
    """
    Get the content parts of an audio file and generate a request.
    :return:
    """
    audio_file_path = "./../data/audio_example.mp3"
    gemini_client = GeminiClient()
    audio_file = AudioFile(audio_file_path, gemini_client=gemini_client)
    # audio_file.delete()
    prompt = ["Listen this audio:", audio_file, "Describe what you heard"]
    model_name = "models/gemini-1.5-pro-latest"
    tokens_count = gemini_client.count_tokens(model_name, prompt)
    print(f"Tokens count: {tokens_count}")
    response = gemini_client.generate_response(model_name, prompt, stream=True)
    for message_chunk in response:
        if message_chunk.parts:
            print(message_chunk.text)


if __name__ == "__main__":
    chat_wit_your_audios()

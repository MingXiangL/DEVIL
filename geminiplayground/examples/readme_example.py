from geminiplayground.core import GeminiClient
from geminiplayground.parts import VideoFile, ImageFile
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

gemini_client = GeminiClient()

video_file_path = "./../data/BigBuckBunny_320x180.mp4"
video_file = VideoFile(video_file_path, gemini_client=gemini_client)

image_file_path = "https://upload.wikimedia.org/wikipedia/commons/4/47/PNG_transparency_demonstration_1.png"
image_file = ImageFile(image_file_path, gemini_client=gemini_client)

multimodal_prompt = [
    "See this video",
    video_file,
    "and this image",
    image_file,
    "Explain what you see."
]

response = gemini_client.generate_response("models/gemini-1.5-pro-latest", multimodal_prompt,
                                           generation_config={"temperature": 0.0, "top_p": 1.0})

# Print the response
for candidate in response.candidates:
    for part in candidate.content.parts:
        if part.text:
            print(part.text)

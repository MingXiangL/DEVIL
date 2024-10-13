## Gemini Playground

![Gemini Logo](https://raw.githubusercontent.com/haruiz/geminiplayground/main/images/logo.png)

Gemini Playground provides a Python interface and a user interface to interact with different Gemini model variants.
With Gemini Playground, you can:

* **Engage in conversation with your data either through a simple code API or using the API:** Upload images and videos
  using a simple API and
  generate responses based on your prompts.
* **Chat with your codebase as you do with images, PDFs and audio files:** Ask Gemini to analyze your code, explain its
  functionality, suggest improvements or even write documentation for it.
* **Explore multimodal capabilities:** Combine different data types in your prompts,
  like asking Gemini to describe what's happening in a video and an image simultaneously.

### Features

* **Intuitive API:** The `GeminiClient` class offers a simple and
  easy-to-use interface for interacting with the Gemini API.
* **Multimodal Support:** Upload and use text, images, videos, and
  code in your prompts.
* **File Management:** Upload, list, and remove files from your
  Gemini storage.
* **Token Counting:** Estimate the number of tokens required for a
  prompt and response.
* **Response Generation:** Generate responses from Gemini based on
  your prompts and uploaded content.
* **Rich Logging:** Get informative and colorful logging messages for
  better understanding of the process.

You can find usage examples in the `examples` directory.

### Installation

```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ geminiplayground
```

### Usage

1. **Set up your API key:**
    * Obtain an API key from Google AI-Studio.
    * Set the `AISTUDIO_API_KEY` environment variable with your API
      key.

2. **Create a `GeminiClient` instance:**

```python
from geminiplayground.core import GeminiClient
from geminiplayground.parts import VideoFile, ImageFile

gemini_client = GeminiClient()
```

3. **Define your files:**

```python
video_file_path = "./data/BigBuckBunny_320x180.mp4"
video_file = VideoFile(video_file_path, gemini_client=gemini_client)

image_file_path = "https://upload.wikimedia.org/wikipedia/commons/4/47/PNG_transparency_demonstration_1.png"
image_file = ImageFile(image_file_path, gemini_client=gemini_client)
```

4. **Create a prompt:**

```python
multimodal_prompt = [
    "See this video",
    video_file,
    "and this image",
    image_file,
    "Explain what you see."
]
```

5. **Generate a response:**

```python
response = gemini_client.generate_response("models/gemini-1.5-pro-latest", multimodal_prompt,
                                           generation_config={"temperature": 0.0, "top_p": 1.0})
# Print the response
for candidate in response.candidates:
    for part in candidate.content.parts:
        if part.text:
            print(part.text)
```

```text
The video is a short animated film called "Big Buck Bunny." It is a comedy about a large, white rabbit 
who is bullied by three smaller animals. The rabbit eventually gets revenge on his tormentors. The film 
was created using Blender, a free and open-source 3D animation software.

The image is of four dice, each a different color. The dice are transparent and have white dots. The 
image is isolated on a black background.
```

6. **You can also chat with your data:**

**Chat with your codebase:**

```python
from rich import print

from geminiplayground.core import GeminiClient
from geminiplayground.parts import GitRepo
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())


def chat_wit_your_code():
    """
    Get the content parts of a github repo and generate a request.
    :return:
    """
    repo = GitRepo.from_url(
        "https://github.com/karpathy/ng-video-lecture",
        branch="master",
        config={
            "content": "code-files",  # "code-files" or "issues"
            "file_extensions": [".py"],
        },
    )
    prompt = [
        "use this codebase:",
        repo,
        "Describe the `bigram.py` file, and generate some code snippets",
    ]
    model = "models/gemini-1.5-pro-latest"
    gemini_client = GeminiClient()
    tokens_count = gemini_client.count_tokens(model, prompt)
    print("Tokens count: ", tokens_count)
    response = gemini_client.generate_response(model, prompt, stream=True)

    # Print the response
    for message_chunk in response:
        if message_chunk.parts:
            print(message_chunk.text)


if __name__ == "__main__":
    chat_wit_your_code()

```

**Chat with your videos:**

```python
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
```

**Chat with your images:**

```python
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
```

**Chat with your Pdfs:**

```python
from rich import print

from geminiplayground.core import GeminiClient
from geminiplayground.parts import PdfFile
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())


def chat_wit_your_pdf():
    """
    Get the content parts of a pdf file and generate a request.
    :return:
    """
    gemini_client = GeminiClient()
    pdf_file_path = "https://www.tnstate.edu/faculty/fyao/COMP3050/Py-tutorial.pdf"
    pdf_file = PdfFile(pdf_file_path, gemini_client=gemini_client)

    prompt = ["Please create a summary of the pdf file:", pdf_file]
    model_name = "models/gemini-1.5-pro-latest"
    tokens_count = gemini_client.count_tokens(model_name, prompt)
    print(f"Tokens count: {tokens_count}")
    response = gemini_client.generate_response(model_name, prompt, stream=True)
    for message_chunk in response:
        if message_chunk.parts:
            print(message_chunk.text)


if __name__ == "__main__":
    chat_wit_your_pdf()
```

**Function calling in chat:**

```python
from dotenv import load_dotenv, find_dotenv

from geminiplayground.core import GeminiPlayground, Message, ToolCall
from geminiplayground.parts import ImageFile

load_dotenv(find_dotenv())

if __name__ == "__main__":
    playground = GeminiPlayground(
        model="models/gemini-1.5-flash-latest"
    )


    @playground.tool
    def subtract(a: int, b: int) -> int:
        """This function only subtracts two numbers"""
        return a - b


    @playground.tool
    def write_poem() -> str:
        """write a poem"""
        return "Roses are red, violets are blue, sugar is sweet, and so are you."


    chat = playground.start_chat(history=[])
    while True:
        user_input = input("You: ")
        if user_input == "exit":
            print(chat.history)
            break
        try:
            model_response = chat.send_message(user_input, stream=True)
            for response_chunk in model_response:
                if isinstance(response_chunk, ToolCall):
                    print(
                        f"Tool: {response_chunk.tool_name}, "
                        f"Result: {response_chunk.tool_result}"
                    )
                    continue
                print(response_chunk.text, end="")
            print()
        except Exception as e:
            print("Something went wrong: ", e)
            break

```

This is a basic example. Explore the codebase and documentation for more
advanced functionalities and examples.

### GUI

You can also use the GUI to interact with Gemini.
Remember to set the `AISTUDIO_API_KEY` environment variable with your API key. You can do so globally, pass it as an
argument to the command, or create a `.env` file in the root of your project and set the `AISTUDIO_API_KEY` variable
there.

For running the GUI, use the following command:

```bash
geminiplayground ui
```

or

```bash
geminiplayground ui --api-key YOUR_API_KEY
```

This will start a local server and open the GUI in your default browser.

![Gemini GUI](https://raw.githubusercontent.com/haruiz/geminiplayground/main/images/ui.png)

To access the uploaded files from the UI, just type `@`. It will open a popup list where you can select the file you
want.

### Contributing

Contributions are welcome! Please see the 'CONTRIBUTING.md` file for guidelines [Coming soon].

### License

This codebase is licensed under the MIT License. See the`LICENSE` file for details. 


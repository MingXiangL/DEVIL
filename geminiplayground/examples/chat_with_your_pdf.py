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

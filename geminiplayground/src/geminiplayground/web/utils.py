from sqlalchemy import select

from geminiplayground.parts import MultimodalPartFactory, GitRepo
from geminiplayground.utils import LibUtils
from geminiplayground.web.db.models import MultimodalPartEntry
from geminiplayground.web.db.session_manager import get_db_session

from asyncio import gather


async def get_parts_from_prompt_text(prompt):
    """
    Transform prompt into parts.
    :param prompt: The prompt text.
    :return: A list of parts.
    """

    prompt_parts = LibUtils.split_and_label_prompt_parts_from_string(prompt)
    files_dir = LibUtils.get_lib_home()
    repos_dir = files_dir.joinpath("repos")

    # Separate multimodal parts for concurrent processing
    text_parts = [part for part in prompt_parts if part["type"] == "text"]
    multimodal_parts = [part for part in prompt_parts if part["type"] == "multimodal"]

    # Process text parts
    parts = [part["value"] for part in text_parts]

    # Process multimodal parts concurrently
    multimodal_results = await gather(
        *[
            process_multimodal_part(part, files_dir, repos_dir)
            for part in multimodal_parts
        ]
    )
    for result in multimodal_results:
        parts.extend(result)

    return parts


async def process_multimodal_part(part, files_dir, repos_dir):
    """
    Process a multimodal part asynchronously.
    :param part: The part to process.
    :param files_dir: The directory for files.
    :param repos_dir: The directory for repos.
    :return: A list of content parts from the multimodal part.
    """
    parts = []
    async for session in get_db_session():
        part_entry = await session.execute(
            select(MultimodalPartEntry).filter(
                MultimodalPartEntry.name == part["value"]
            )
        )
        part_entry = part_entry.scalars().first()
        if part_entry:
            content_type = part_entry.content_type
            if content_type in ["image", "video", "audio", "pdf"]:
                file_path = files_dir.joinpath(part_entry.name)
                multimodal_part = MultimodalPartFactory.from_path(file_path)
                parts.extend(multimodal_part.content_parts())
            elif content_type == "repo":
                repo_folder = repos_dir.joinpath(part_entry.name)
                repo = GitRepo.from_folder(
                    repo_folder,
                    config={"content": "code-files", "file_extensions": [".py"]},
                )
                parts.extend(repo.content_parts())
            else:
                raise ValueError(f"Unsupported content type: {content_type}")
    return parts

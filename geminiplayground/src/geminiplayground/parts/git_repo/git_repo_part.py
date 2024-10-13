import logging
import typing
from pathlib import Path
from urllib.parse import urlparse

from github import Github
import git
from geminiplayground.core import GeminiClient
from geminiplayground.utils import (
    GitUtils,
    GitRemoteProgress,
    LibUtils,
)
from ..multimodal_part import MultimodalPart
import codecs

logger = logging.getLogger("rich")


class GitRepoBranchNotFoundException(Exception):
    pass


class GitRepo(MultimodalPart):
    """
    Git Repo Part implementation
    """

    def __init__(self, repo_folder: typing.Union[str, Path], **kwargs):
        # set the output directory for the repos
        try:
            repo_folder = Path(repo_folder).resolve(strict=True)
            logger.info(f"Checking if {repo_folder} is a git repository")

            assert repo_folder.exists(), f"{repo_folder} does not exist"
            assert GitUtils.folder_contains_git_repo(
                repo_folder
            ), f"{repo_folder} is not a git repository"

            self.repo_folder = repo_folder
            self.repo = git.Repo(repo_folder)
            self.gemini_client = kwargs.get("gemini_client", GeminiClient())

            self.config = kwargs.setdefault("config", {"content": "code-files"})
            self.content = self.config.get("content", "code-files")

            logger.info(f"Repo folder: {self.repo_folder}")
            logger.info(f"Content: {self.content}")
            logger.info(f"Config: {self.config}")

            valid_contents = {"code-files", "issues"}
            if self.content not in valid_contents:
                raise ValueError(
                    f"Invalid content {self.content}, should be code-files or issues"
                )
        except Exception as e:
            logger.error(e)
            raise e

    @classmethod
    def from_folder(cls, folder: typing.Union[str, Path], **kwargs):
        """
        Create a GitRepo instance from a folder
        :param folder: the folder to create the GitRepo instance from
        :param kwargs: additional arguments to pass to the GitRepo constructor
        :return:
        """

        return cls(folder, **kwargs)

    @classmethod
    def from_url(cls, repo_url: str, branch: str = "main", **kwargs):
        """
        Create a GitRepo instance from a repo url
        :param repo_url: the url of the repo to create the GitRepo instance from
        :param branch: the branch to clone the repo from
        :param kwargs: additional arguments to pass to the GitRepo constructor
        :return:
        """
        playground_home = LibUtils.get_lib_home()
        default_repos_folder = playground_home.joinpath("repos")
        repos_folder = kwargs.get("repos_folder", default_repos_folder)
        repos_folder = Path(repos_folder)
        repos_folder.mkdir(parents=True, exist_ok=True)

        repo_name = GitUtils.get_repo_name_from_url(repo_url)
        repo_folder = repos_folder.joinpath(repo_name)
        repo_folder.mkdir(parents=True, exist_ok=True)

        if not GitUtils.check_github_repo_branch_exists(repo_url, branch):
            available_branches = GitUtils.get_github_repo_available_branches(repo_url)
            error_message = (
                f"Branch {branch} does not exist in {repo_url}. "
                f"Available branches for the repository {repo_name} are: {available_branches}"
            )
            logger.error(error_message)
            raise GitRepoBranchNotFoundException(error_message)

        folder_is_empty = not any(repo_folder.iterdir())
        if folder_is_empty:
            try:
                git.Repo.clone_from(
                    url=repo_url,
                    to_path=repo_folder,
                    branch=branch,
                    progress=GitRemoteProgress(),
                )
            except Exception as e:
                logger.error(e)
                raise e
        config = kwargs.setdefault("config", {"content": "code-files"})
        print(repo_folder, config)
        return cls(repo_folder, config=config)

    def __get_parts_from_code_files(self):
        """
        Get the code parts from the repo
        :return:
        """
        file_extensions = self.config.get("file_extensions", None)
        exclude_dirs = self.config.get("exclude_dirs", None)

        code_files = GitUtils.get_code_files_in_dir(
            self.repo_folder, file_extensions, exclude_dirs
        )
        parts = []
        for file in code_files:
            with codecs.open(file, "r", encoding="utf-8", errors="ignore") as f:
                code_content = f.read()
                parts.append(
                    f"""
                file: {file}
                ```python
                {code_content}
                ```
                """
                )
        return parts

    def __get_parts_from_repos_issues(self):
        """
        Get the issues from the repo
        :return:
        """
        issues_state = self.config.get("issues_state", "open")

        remotes = self.repo.remotes
        assert len(remotes) > 0, "No remotes found"
        remote = remotes[0]
        url = remote.url
        g = Github()
        repo_path = urlparse(url).path[1:]
        remote_repo = g.get_repo(repo_path)
        issues = remote_repo.get_issues(state=issues_state)
        parts = []
        for issue in issues:
            parts.append(
                f"""
            issue: {issue.title}
            {issue.body}
            """
            )
        return parts

    def content_parts(self):
        """
        Get the content parts for the repo
        :return:
        """
        try:
            functions_map = {
                "code-files": self.__get_parts_from_code_files,
                "issues": self.__get_parts_from_repos_issues,
            }
            return functions_map[self.content]()
        except Exception as e:
            logger.error(e)
            raise e

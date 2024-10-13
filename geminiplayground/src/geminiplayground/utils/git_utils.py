import subprocess
import typing
from pathlib import Path
import git
import validators


class GitUtils:
    """
    Git utilities
    """

    @staticmethod
    def get_code_files_in_dir(
            root_dir: typing.Union[str, Path], files_extensions=None, exclude_dirs=None
    ) -> list:
        """
        Extract code files from the repo
        :return:
        """
        default_exclude_dirs = [
            ".git",
            "node_modules",
            ".venv",
            "__pycache__",
            ".idea",
            ".vscode",
            "build",
            "dist",
            "target",
        ]
        ignore_dirs = default_exclude_dirs
        if exclude_dirs is not None:
            ignore_dirs += exclude_dirs

        if files_extensions is None:
            files_extensions = [
                ".py",
                ".java",
                ".cpp",
                ".h",
                ".c",
                ".go",
                ".js",
                ".html",
                ".css",
                ".sh",
            ]
        code_files = []
        for path in Path(root_dir).rglob("*"):
            if path.is_file() and path.suffix in files_extensions:
                # Check if any part of the path is in the ignore list
                if not any([ignore_dir in path.parts for ignore_dir in ignore_dirs]):
                    code_files.append(path)

        return code_files

    @staticmethod
    def folder_contains_git_repo(path):
        """
        Check if a given folder is a git repository
        :param path:
        :return: True if the given folder is a repor or false otherwise
        """
        try:
            _ = git.Repo(path).git_dir
            return True
        except (git.exc.InvalidGitRepositoryError, Exception):
            return False

    @staticmethod
    def get_repo_name_from_url(url: str) -> str:
        """
        Get and return the repo name from a valid github url
        :rtype: str
        """
        last_slash_index = url.rfind("/")
        last_suffix_index = url.rfind(".git")
        if last_suffix_index < 0:
            last_suffix_index = len(url)
        if last_slash_index < 0 or last_suffix_index <= last_slash_index:
            raise Exception("invalid url format {}".format(url))
        return url[last_slash_index + 1: last_suffix_index]

    @classmethod
    def get_repo_name_from_path(cls, path: str) -> str:
        """
        Get and return the repo name from a valid github url
        :rtype: str
        """
        assert cls.folder_contains_git_repo(path), "Invalid git repo path"
        return Path(path).name

    @classmethod
    def get_repo_name(cls, path: str) -> str:
        """
        Get the repo name from a path
        :param path:
        :return:
        """
        if validators.url(path):
            return cls.get_repo_name_from_url(path)
        else:
            return cls.get_repo_name_from_path(path)

    @staticmethod
    def get_github_repo_available_branches(remote_url):
        """
        Get the available branches in a github repository
        :param remote_url:
        :return:
        """
        branches = subprocess.check_output(["git", "ls-remote", "--heads", remote_url])
        branches = branches.decode("utf-8").strip().split("\n")
        branches = [branch.split("refs/heads/")[1] for branch in branches]
        return branches

    @classmethod
    def check_github_repo_branch_exists(cls, remote_url, branch_name):
        """
        Check if a branch exists in a github repository
        """
        # List all branches from the remote repository
        branches = cls.get_github_repo_available_branches(remote_url)

        # Check if the specified branch exists
        return branch_name in branches

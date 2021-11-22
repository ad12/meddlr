import logging
import os
import shutil
from abc import ABC, abstractmethod
from typing import Any, Optional

from iopath.common.file_io import PathHandler

import meddlr as mr
from meddlr.utils import env
from meddlr.utils.cluster import Cluster

_REPO_DIR = os.path.join(os.path.dirname(__file__), "../..")
_LOGGER = logging.getLogger(__name__)

__all__ = ["GeneralPathHandler", "GithubHandler"]


class GeneralPathHandler(PathHandler, ABC):
    PREFIX = ""

    def _get_supported_prefixes(self):
        return [self.PREFIX]

    def _get_local_path(self, path: str, **kwargs):
        name = path[len(self.PREFIX) :]
        return os.path.join(self._root_dir(), name)

    def _open(self, path, mode="r", **kwargs):
        return env.get_path_manager().open(self._get_local_path(path), mode, **kwargs)

    def _mkdirs(self, path: str, **kwargs):
        os.makedirs(self._get_local_path(path), exist_ok=True)

    @abstractmethod
    def _root_dir(self) -> str:
        pass


class DataHandler(GeneralPathHandler):
    PREFIX = "data://"

    def _root_dir(self):
        return Cluster.working_cluster().get_path("data_dir")


class ResultsHandler(GeneralPathHandler):
    PREFIX = "results://"

    def _root_dir(self):
        return Cluster.working_cluster().get_path("results_dir")


class CacheHandler(GeneralPathHandler):
    PREFIX = "cache://"

    def _root_dir(self):
        return Cluster.working_cluster().get_path("cache_dir")


class GithubHandler(GeneralPathHandler):
    """Handler to download, cache, and match Github files to local folder structure.

    Often, it may be useful to reference files from the meddlr Github repository.
    This class (when used with :cls:`iopath.common.file_io.PathHandler`) fetches
    and locally caches files from the Meddlr Github repository.

    This class fetches files from the most recent tagged version in Meddlr.

    Attributes:
        github_url (str): The base URL for the Github repository.
        default_branch_or_tag (str): The default branch or tag to download files from.

    Examples:
        >>> gh = GithubHandler(github_url="https://github.com/ad12/meddlr")
        >>> gh.get_local_path("github://annotations")
    """

    PREFIX = "github://"

    def __init__(
        self,
        github_url: str = None,
        default_branch_or_tag: str = "main",
        default_cache_path: str = None,
        async_executor=None,
    ) -> None:
        if github_url is None:
            github_url = env.get_github_url()
        self.github_url = github_url
        self.default_branch_or_tag = default_branch_or_tag
        self.default_cache_path = default_cache_path
        super().__init__(async_executor=async_executor)

    def _root_dir(self):
        return None

    def _get_default_branch_or_tag(self):
        return mr.__version__

    def _get_default_cache_path(self):
        return os.path.join(
            env.get_path_manager().get_local_path(Cluster.working_cluster().cache_dir),
            f"github-repo/v{mr.__version__}",
        )

    def download(self, cache_dir, branch_or_tag: str = None, force: bool = False):
        if branch_or_tag is None:
            branch_or_tag = self.default_branch_or_tag
        if branch_or_tag is None:
            branch_or_tag = self._get_default_branch_or_tag()
        return download_github_repository(
            url=self.github_url, branch_or_tag=branch_or_tag, cache_path=cache_dir, force=force
        )

    def _get_local_path(
        self,
        path: str,
        branch_or_tag: str = None,
        force: bool = False,
        cache_dir: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """Get local path to GitHub repository for the current meddlr version.

        This implementation will only download the GitHub repository if it does
        not exist. To force a download, set ``force=True``.

        Args:
            path (str): The relative file path in the GitHub repository.
                Must start with ``'github://'``.
            branch_or_tag (str): The Github branch or tag to download from.
            force (bool, optional): If ``True``, force a download of the Github
                repository. Defaults to ``False``.
            cache_dir (str, optional): The local cache directory to use.

        Returns:
            str: The local path to the file/directory.
        """
        path = path[len(self.PREFIX) :]
        self._check_kwargs(kwargs)
        if cache_dir is None:
            cache_dir = (
                self.default_cache_path
                if self.default_cache_path
                else self._get_default_cache_path()
            )

        if force or not os.path.exists(cache_dir) or not os.listdir(cache_dir):
            logger = logging.getLogger(__name__)
            logger.info(
                "Downloading github repository (tag/branch {}) from {}...".format(
                    mr.__version__, self.github_url
                )
            )
            self.download(cache_dir, force=force, branch_or_tag=branch_or_tag)
            logger.info("Repository cached to {}".format(cache_dir))

        expected_path = os.path.join(str(cache_dir), path)
        if not os.path.exists(expected_path):
            raise FileNotFoundError(
                "Could not find {} in local Github cache: {}".format(path, cache_dir)
            )
        return expected_path


class AnnotationsHandler(GeneralPathHandler):
    PREFIX = "ann://"

    def _get_cache_path(self):
        return "github://annotations"

    def _root_dir(self):
        local_path = os.path.abspath(os.path.join(_REPO_DIR, "annotations"))
        if os.path.isdir(local_path):
            return local_path

        cache_path = env.get_path_manager().get_local_path(self._get_cache_path())
        if not os.path.isdir(cache_path):
            self.download()
        return cache_path

    def download(self):
        """Downloads annotations from meddlr github.

        Note:
            This method downloads the full repository of the current
            meddlr version and then copies the annotation files to
            the respective directory. This may not be very efficient,
            but it ensures that the annotation download process is
            less onerous on the user.
        """
        _LOGGER.info("Downloading annotations...")
        repo_path = download_github_repository()
        cache_path = env.get_path_manager().get_local_path(self._get_cache_path())
        os.makedirs(cache_path, exist_ok=True)
        retval = os.system(f"cp -r {repo_path}/annotations {cache_path}/")
        if retval != 0:
            raise RuntimeError("Could not download annotations.")


def download_github_repository(url, cache_path, branch_or_tag="main", force=False) -> str:
    """Downloads the repository from Github.

    Args:
        version (str): The version to download. Defaults to the
            version of the current codebase.
        path (str): The path to download the repository to.
            Defaults to the the cache directory.
        force (bool): Whether to overwrite existing files.

    Returns:
        str: The path to the downloaded repository.
    """
    url = f"{url}/tarball/{branch_or_tag}"
    cache_path = str(cache_path)
    cache_path = cache_path.format(branch_or_tag=branch_or_tag)

    dir_exists_and_not_empty = os.path.isdir(cache_path) and os.listdir(cache_path)
    if dir_exists_and_not_empty:
        if not force:
            return cache_path
        shutil.rmtree(cache_path)
    os.makedirs(cache_path, exist_ok=True)

    retval = os.system(f"wget -O - {url} | tar xz -C {cache_path}")
    if retval != 0:
        raise RuntimeError("Could not download repository.")

    folders = os.listdir(cache_path)
    assert len(folders) == 1
    curr_path = os.path.join(cache_path, folders[0])
    retval = os.system(f"mv {curr_path}/* {cache_path}")
    if retval != 0:
        raise RuntimeError("Could not download repository.")
    shutil.rmtree(curr_path)

    return cache_path


_path_manager = env.get_path_manager()
_path_manager.register_handler(DataHandler())
_path_manager.register_handler(ResultsHandler())
_path_manager.register_handler(CacheHandler())
_path_manager.register_handler(GithubHandler(env.get_github_url(), default_branch_or_tag=None))
_path_manager.register_handler(AnnotationsHandler())
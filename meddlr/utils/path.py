import io
import logging
import os
import shutil
import weakref
from abc import ABC, abstractmethod
from typing import Any, Optional

from iopath.common.file_io import PathHandler, PathManager

import meddlr as mr
from meddlr.utils import env
from meddlr.utils.cluster import Cluster

try:
    import gdown

    _GDOWN_AVAILABLE = True
except ImportError:
    _GDOWN_AVAILABLE = False

try:
    import iocursor

    _IOCURSOR_AVAILABLE = True
except ImportError:
    _IOCURSOR_AVAILABLE = False

try:
    import requests

    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False

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
        return f"v{mr.__version__}"

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


class GoogleDriveHandler(GeneralPathHandler):
    """Handler to download, cache, and match Google drive files to local folder structure.

    Publicly available files on google drive can be downloaded using this handler.
    By default, files are cached to "~/.cache/gdrive/<file id>".

    Examples:
        >>> handler = GoogleDriveHandler()
        >>> handler.get_local_path("gdrive://https://drive.google.com/file/d/14VQf4esuZVy_Xf6IUciBas81j0JpaqCb/view?usp=sharing")  # recommended  # noqa: E501
        >>> # OR
        >>> handler.get_local_path("gdrive://14VQf4esuZVy_Xf6IUciBas81j0JpaqCb")

    Note:
        This handler requires ``gdown`` to be installed. Install it with ``pip install gdown``.
    """

    PREFIX = "gdrive://"

    def __init__(self, cache_dir=None, **kwargs) -> None:
        if cache_dir is None:
            cache_dir = os.path.expanduser("~/.cache/gdrive")
        self.cache_dir = cache_dir
        super().__init__(**kwargs)

    def _root_dir(self):
        return None

    def _get_local_path(
        self,
        path: str,
        force: bool = False,
        cache: Optional[str] = None,
        is_folder=None,
        **kwargs: Any,
    ) -> str:
        """Get local path to google drive file.

        To force a download, set ``force=True``.

        Args:
            path (str): The relative file path in the GitHub repository.
                Must start with ``'gdrive://'``.
            force (bool, optional): If ``True``, force a download of the Github
                repository. Defaults to ``False``.
            cache (str, optional): The path to cache file to.

        Returns:
            str: The local path to the file/directory.
        """
        if not _GDOWN_AVAILABLE:
            raise ModuleNotFoundError("`gdown` not installed. Install it via `pip install gdown`")

        path = str(path)
        path = path[len(self.PREFIX) :]
        self._check_kwargs(kwargs)

        if is_folder is None:
            is_folder = "drive.google.com" in path and "folders" in path

        if is_folder:
            path = self._handle_folder(path, cache, force)
        else:
            path = self._handle_file(path, cache, force)
        return path

    def _handle_file(self, path: str, cache: str, force: bool) -> str:
        if "drive.google.com" in path:
            gdrive_id = path.split("/d/")[1].split("/")[0]
        else:
            gdrive_id = path

        if cache is None:
            cache = os.path.join(self.cache_dir, gdrive_id)
        else:
            cache = str(cache)
        os.makedirs(os.path.dirname(cache), exist_ok=True)

        if force or not os.path.exists(cache):
            logger = logging.getLogger(__name__)
            logger.info("Downloading gdrive file from {}...".format(path))
            gdown.download(id=gdrive_id, output=cache)
            logger.info("File cached to {}".format(cache))

        return cache

    def _handle_folder(self, path: str, cache: str, force: bool) -> str:
        from gdown.download_folder import client, parse_google_drive_file

        logger = logging.getLogger(__name__)

        if cache is None:
            folder_page = client.get(path)
            if folder_page.status_code != 200:
                raise ValueError("Unable to download gdrive url: {}".format(path))
            gdrive_file, _ = parse_google_drive_file(path, folder_page.content)
            cache = os.path.join(self.cache_dir, gdrive_file.name)
        else:
            cache = str(cache)

        if force or not os.path.exists(cache):
            logger.info("Downloading gdrive folder from {}...".format(path))
            gdown.download_folder(url=path, output=cache)
            logger.info("Folder cached to {}".format(cache))
        else:
            logger.info("Folder {} already exists. Skipping download.".format(cache))

        return cache


class URLHandler(GeneralPathHandler):
    """Handler to download, cache, and match HuggingFace files to local folder structure.

    Publicly available files on HuggingFace can be downloaded using this handler.
    By default, files are not cached.

    Examples:

    .. code-block:: python

        handler = URLHandler()
        >>> handler.get_local_path("wget://https://huggingface.co/arjundd/vortex-release/blob/main/mridata_knee_3dfse/Supervised/config.yaml")  # noqa: E501
    """

    # DO NOT REORDER.
    PREFIX = ("https://", "http://")
    LEGACY_PREFIX = ("download://",)

    # Certain URLs have to be redirected to a different handler.
    _DOMAIN_TO_PREFIX_MATCH = {"drive.google.com": "gdrive://"}

    def __init__(self, path_manager: PathManager, cache_dir=None, **kwargs) -> None:
        self.path_manager = weakref.proxy(path_manager)
        self.cache_dir = cache_dir
        super().__init__(**kwargs)

    def _get_supported_prefixes(self):
        return self.PREFIX + self.LEGACY_PREFIX

    def _root_dir(self):
        return None

    def _check_other_prefixes(self, path: str) -> str:
        path = str(path)
        for domain, prefix in self._DOMAIN_TO_PREFIX_MATCH.items():
            if domain in path:
                path = self._parse_legacy_prefix(path)
                return f"{prefix}{path}"
        return False

    def _parse_legacy_prefix(self, path: str) -> str:
        """Parses legacy prefixes out of the url path."""
        for prefix in self.LEGACY_PREFIX:
            if path.startswith(prefix):
                return path[len(prefix) :]
        return path

    def _open(self, path, mode="r", **kwargs):
        if mode not in ("r", "rb"):
            raise ValueError("Only file reading is supported.")

        other_domain_path = self._check_other_prefixes(path)
        if other_domain_path:
            return self.path_manager.open(other_domain_path, mode=mode, **kwargs)

        if not _REQUESTS_AVAILABLE:
            raise ModuleNotFoundError(
                "`requests` not installed. Install it via `pip install requests`"
            )

        # Buffering is not a supported argument for requests.
        kwargs.pop("buffering", None)

        path = self._parse_legacy_prefix(path)
        r = requests.get(path, **kwargs)
        r.raise_for_status()

        if _IOCURSOR_AVAILABLE:
            return iocursor.Cursor(r.content)
        else:
            return io.BytesIO(r.content)

    def _get_local_path(
        self,
        path: str,
        force: bool = False,
        cache: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """Get local path to google drive file.

        To force a download, set ``force=True``.

        Args:
            path (str): The relative file path in the GitHub repository.
                Must start with ``'gdrive://'``.
            force (bool, optional): If ``True``, force a download of the Github
                repository. Defaults to ``False``.
            cache (str, optional): The path to cache file to.

        Returns:
            str: The local path to the file/directory.
        """
        other_domain_path = self._check_other_prefixes(path)
        if other_domain_path:
            return self.path_manager.get_local_path(
                other_domain_path, force=force, cache=cache, **kwargs
            )

        if not _REQUESTS_AVAILABLE:
            raise ModuleNotFoundError(
                "`requests` not installed. Install it via `pip install requests`"
            )

        path = self._parse_legacy_prefix(str(path))
        self._check_kwargs(kwargs)

        if cache is None:
            cache = os.path.join(
                env.get_path_manager().get_local_path(Cluster.working_cluster().cache_dir), "url"
            )
            base = path.split("://", 1)[1]
            out_file = os.path.join(cache, base)
        elif str(cache).endswith(os.pathsep) or os.path.isdir(path):
            out_file = os.path.join(cache, os.path.basename(path))
        else:
            out_file = cache

        # TODO: Make caching smarter based on time of file creation and update.
        if force or not os.path.isfile(out_file):
            logger = logging.getLogger(__name__)
            logger.info("Downloading file from {}...".format(path))
            r = requests.get(path)
            r.raise_for_status()
            os.makedirs(os.path.dirname(out_file), exist_ok=True)
            with open(out_file, "wb") as f:
                f.write(r.content)
            logger.info("File cached to {}".format(out_file))

        return out_file


class DownloadHandler(GeneralPathHandler):
    """Handler to download files from a URL.

    Currently this is limited to downloading from Google Drive.

    Examples:
        >>> handler = DownloadHandler()
        >>> handler.get_local_path("download://https://drive.google.com/file/d/1fWgHNUljPrJj-97YPbbrqugSPnS2zXnx/view?usp=sharing")  # noqa: E501
    """

    PREFIX = "download://"

    def __init__(self, path_manager: PathManager, async_executor=None) -> None:
        # Avoid memory leak.
        self.path_manager = weakref.proxy(path_manager)
        super().__init__(async_executor=async_executor)

    def _root_dir(self):
        return None

    def _get_local_path(
        self,
        path: str,
        **kwargs: Any,
    ) -> str:
        """Get local path to google drive file.

        To force a download, set ``force=True``.

        Args:
            path (str): The relative file path in the GitHub repository.
                Must start with ``'download://'``.

        Returns:
            str: The local path to the file/directory.
        """
        path = path[len(self.PREFIX) :]

        if "drive.google.com" in path:
            return self.path_manager.get_local_path(f"gdrive://{path}", **kwargs)
        elif "huggingface.co" in path:
            return self.path_manager.get_local_path(path, **kwargs)
        else:
            raise ValueError(f"Download not supported for url {path}")


class ForceDownloadHandler(DownloadHandler):
    """Like :cls:`DownloadHandler`, but always downloads (even if it is cached).

    If the file is cached, it will be replaced by the downloaded version.

    Examples:
        >>> handler = ForceDownloadHandler()
        >>> handler.get_local_path("force-download://https://drive.google.com/file/d/1fWgHNUljPrJj-97YPbbrqugSPnS2zXnx/view?usp=sharing")  # noqa: E501
    """

    PREFIX = "force-download://"

    def _get_local_path(
        self,
        path: str,
        **kwargs: Any,
    ) -> str:
        path = f"{DownloadHandler.PREFIX}{path[len(self.PREFIX) :]}"
        kwargs.pop("force", None)
        return self.path_manager.get_local_path(path, force=True, **kwargs)


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
_path_manager.register_handler(GoogleDriveHandler())
_path_manager.register_handler(ForceDownloadHandler(_path_manager))
_path_manager.register_handler(URLHandler(_path_manager))

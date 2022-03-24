import os
import time

import yaml

from meddlr.utils import env
from meddlr.utils.path import (
    ForceDownloadHandler,
    GithubHandler,
    GoogleDriveHandler,
    URLHandler,
    download_github_repository,
)

# def test_meddlr_path_manager():
#     pm = env.get_path_manager()

#     # Annotations
#     assert os.path.isdir(pm.get_local_path("ann://"))

#     # Github
#     assert os.path.isdir(pm.get_local_path("github://"))


def test_download_github(tmpdir):
    download_dir = tmpdir.mkdir("download")
    download_github_repository(env.get_github_url(), branch_or_tag="main", cache_path=download_dir)
    assert os.path.isdir(download_dir.join("annotations").strpath)


def test_github_handler(tmpdir):
    download_dir = tmpdir.mkdir("download")
    handler = GithubHandler(
        env.get_github_url(), default_branch_or_tag="main", default_cache_path=download_dir / "main"
    )
    path = handler._get_local_path("github://annotations")
    assert str(path) == os.path.join(download_dir.strpath, "main", "annotations")
    assert os.path.isdir(path)


def test_gdrive_handler(tmpdir):
    download_dir = tmpdir.mkdir("download")

    handler = GoogleDriveHandler(cache_dir=tmpdir)

    # File
    url = "gdrive://https://drive.google.com/file/d/1fWgHNUljPrJj-97YPbbrqugSPnS2zXnx/view?usp=sharing"  # noqa: E501
    cache = download_dir / "hello-world.txt"
    path = handler._get_local_path(url, cache=cache)
    assert os.path.exists(path)
    mtime = os.path.getmtime(path)

    path = handler._get_local_path(url, cache=cache)
    mtime2 = os.path.getmtime(path)
    assert mtime2 == mtime

    # Folder
    folder_url = "gdrive://https://drive.google.com/drive/folders/1UosSskt3H61wcIGUNehhsYoHNBmk-bGi?usp=sharing"  # noqa: E501
    path = handler._get_local_path(folder_url)
    assert os.path.isdir(path)
    mtime = os.path.getmtime(path)

    path = handler._get_local_path(folder_url)
    mtime2 = os.path.getmtime(path)
    assert mtime2 == mtime

    cache = download_dir / "sample-dir"
    path = handler._get_local_path(folder_url, cache=cache)
    assert os.path.isdir(cache)


def test_force_download(tmpdir):
    download_dir = tmpdir.mkdir("download")
    url = "force-download://https://huggingface.co/datasets/arjundd/meddlr-data/resolve/main/test-data/test-exps/basic-cpu.tar.gz"  # noqa: E501
    cache = download_dir / "sample-download.zip"

    path_manager = env.get_path_manager("meddlr_test")
    path_manager.register_handler(URLHandler(path_manager))
    handler = ForceDownloadHandler(path_manager)

    path = handler._get_local_path(url, cache=cache)
    assert os.path.exists(path)
    mtime = os.path.getmtime(path)
    time.sleep(0.1)

    path = handler._get_local_path(url, cache=cache)
    mtime2 = os.path.getmtime(path)

    assert mtime2 != mtime


def test_url_local_path(tmpdir):
    download_dir = tmpdir.mkdir("download")
    url = "https://huggingface.co/arjundd/vortex-release/raw/main/fastmri_brain_mini/Aug_Motion/config.yaml"  # noqa: E501

    path_manager = env.get_path_manager("meddlr_test")
    handler = URLHandler(path_manager, cache_dir=tmpdir)

    cache = download_dir / "config.yaml"
    path = handler._get_local_path(url, cache=cache, force=True)
    assert os.path.exists(path)
    mtime = os.path.getmtime(path)

    path = handler._get_local_path(url, cache=cache)
    mtime2 = os.path.getmtime(path)
    assert mtime2 == mtime


def test_url_open(tmpdir):
    download_dir = tmpdir.mkdir("download")
    url = "https://huggingface.co/arjundd/vortex-release/raw/main/fastmri_brain_mini/Aug_Motion/config.yaml"  # noqa: E501

    path_manager = env.get_path_manager("meddlr_test")
    handler = URLHandler(path_manager, cache_dir=tmpdir)

    cache = download_dir / "config.yaml"
    path = handler._get_local_path(url, cache=cache, force=True)
    with open(path) as f:
        expected = yaml.safe_load(f)

    with handler._open(url) as f:
        out = yaml.safe_load(f)

    assert out == expected

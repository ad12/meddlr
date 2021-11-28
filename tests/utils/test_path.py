import os

from meddlr.utils import env
from meddlr.utils.path import GithubHandler, GoogleDriveHandler, download_github_repository

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
    gdrive_id = "1fWgHNUljPrJj-97YPbbrqugSPnS2zXnx"

    handler = GoogleDriveHandler()

    cache_file = download_dir / "sample-download.zip"
    path = handler._get_local_path(
        f"gdrive://https://drive.google.com/file/d/{gdrive_id}/view?usp=sharing",
        cache_file=cache_file,
    )
    assert os.path.exists(path)

    cache_file = download_dir / "sample-download2.zip"
    path = handler._get_local_path(f"gdrive://{gdrive_id}", cache_file=cache_file)
    assert os.path.exists(path)

import os

from meddlr.utils import env
from meddlr.utils.path import (
    ForceDownloadHandler,
    GithubHandler,
    GoogleDriveHandler,
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
    gdrive_id = "1fWgHNUljPrJj-97YPbbrqugSPnS2zXnx"

    handler = GoogleDriveHandler(cache_dir=tmpdir)

    cache_file = download_dir / "sample-download.zip"
    path = handler._get_local_path(
        f"gdrive://https://drive.google.com/file/d/{gdrive_id}/view?usp=sharing",
        cache_file=cache_file,
        force=True,
    )
    assert os.path.exists(path)
    mtime = os.path.getmtime(path)

    path = handler._get_local_path(
        f"gdrive://https://drive.google.com/file/d/{gdrive_id}/view?usp=sharing",
        cache_file=cache_file,
    )
    mtime2 = os.path.getmtime(path)
    assert mtime2 == mtime

    cache_file = download_dir / "sample-download2.zip"
    path = handler._get_local_path(f"gdrive://{gdrive_id}", cache_file=cache_file)
    assert os.path.exists(path)

    # Folder
    folder_url = "gdrive://https://drive.google.com/drive/folders/1UosSskt3H61wcIGUNehhsYoHNBmk-bGi?usp=sharing"  # noqa: E501
    path = handler._get_local_path(folder_url)
    assert os.path.isdir(path)
    mtime = os.path.getmtime(path)

    path = handler._get_local_path(folder_url)
    mtime2 = os.path.getmtime(path)
    assert mtime2 == mtime

    cache_file = download_dir / "sample-dir"
    path = handler._get_local_path(folder_url, cache_file=cache_file)
    assert os.path.isdir(cache_file)


def test_force_download(tmpdir):
    download_dir = tmpdir.mkdir("download")
    gdrive_id = "1fWgHNUljPrJj-97YPbbrqugSPnS2zXnx"
    cache_file = download_dir / "sample-download.zip"

    path_manager = env.get_path_manager("meddlr_test")
    path_manager.register_handler(GoogleDriveHandler())
    handler = ForceDownloadHandler(path_manager)

    path = handler._get_local_path(
        f"force-download://https://drive.google.com/file/d/{gdrive_id}/view?usp=sharing",
        cache_file=cache_file,
    )
    assert os.path.exists(path)
    mtime = os.path.getmtime(path)

    path = handler._get_local_path(
        f"force-download://https://drive.google.com/file/d/{gdrive_id}/view?usp=sharing",
        cache_file=cache_file,
    )
    mtime2 = os.path.getmtime(path)

    assert mtime2 != mtime

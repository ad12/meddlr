import pytest

from meddlr.config.config import get_cfg


def test_format_fields():
    cfg = get_cfg()
    cfg.DESCRIPTION.BRIEF = 'f"seed={SEED},project={DESCRIPTION.PROJECT_NAME}"'
    cfg.format_fields()
    assert cfg.DESCRIPTION.BRIEF == f"seed={cfg.SEED},project={cfg.DESCRIPTION.PROJECT_NAME}"

    cfg = get_cfg()
    cfg.DESCRIPTION.BRIEF = "f'{SEED},project={DESCRIPTION.PROJECT_NAME}'"
    cfg.format_fields()
    assert cfg.DESCRIPTION.BRIEF == f"{cfg.SEED},project={cfg.DESCRIPTION.PROJECT_NAME}"

    cfg = get_cfg()
    cfg.DESCRIPTION.BRIEF = 'f"{SEED},project={DESCRIPTION.PROJECT_NAME}-today"'
    cfg.format_fields()
    assert cfg.DESCRIPTION.BRIEF == f"{cfg.SEED},project={cfg.DESCRIPTION.PROJECT_NAME}-today"


def test_format_fields_unroll():
    cfg = get_cfg()
    cfg.AUG_TRAIN.UNDERSAMPLE.ACCELERATIONS = (6, 8)
    cfg.DESCRIPTION.BRIEF = (
        'f"{AUG_TRAIN.UNDERSAMPLE.ACCELERATIONS},project={DESCRIPTION.PROJECT_NAME}-today"'
    )
    cfg.format_fields(unroll=True)
    expected = "{},project={}-today".format(
        "-".join(str(x) for x in cfg.AUG_TRAIN.UNDERSAMPLE.ACCELERATIONS),
        cfg.DESCRIPTION.PROJECT_NAME,
    )
    assert cfg.DESCRIPTION.BRIEF == expected


def test_format_fields_nested():
    """Format fields inside lists and tuples."""
    cfg = get_cfg()
    cfg.DESCRIPTION.TAGS = ['f"seed={SEED},project={DESCRIPTION.PROJECT_NAME}"', "foobar"]
    cfg.format_fields()
    assert cfg.DESCRIPTION.TAGS == [
        f"seed={cfg.SEED},project={cfg.DESCRIPTION.PROJECT_NAME}",
        "foobar",
    ]


def test_get_recursive():
    cfg = get_cfg()
    cfg.DESCRIPTION.BRIEF = "foobar"

    assert cfg.get_recursive("DESCRIPTION.BRIEF") == "foobar"
    assert cfg.get_recursive("DESCRIPTION.FOO", None) is None
    with pytest.raises(KeyError):
        cfg.get_recursive("DESCRIPTION.FOO")


def test_get_recursive_with_list():
    cfg = get_cfg()
    cfg.DESCRIPTION.BRIEF = "foobar"

    assert cfg.get_recursive("DESCRIPTION.BRIEF[0]") == "f"
    with pytest.raises(IndexError):
        cfg.get_recursive("DESCRIPTION.BRIEF[1000]")

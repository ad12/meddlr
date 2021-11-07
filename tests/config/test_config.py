import unittest

from meddlr.config.config import get_cfg


class TestConfig(unittest.TestCase):
    def test_format_fields(self):
        cfg = get_cfg()
        cfg.DESCRIPTION.BRIEF = 'f"seed={SEED},project={DESCRIPTION.PROJECT_NAME}"'
        cfg.format_fields()
        assert cfg.DESCRIPTION.BRIEF == f"seed={cfg.SEED},project={cfg.DESCRIPTION.PROJECT_NAME}"

        cfg = get_cfg()
        cfg.DESCRIPTION.BRIEF = 'f"{SEED},project={DESCRIPTION.PROJECT_NAME}"'
        cfg.format_fields()
        assert cfg.DESCRIPTION.BRIEF == f"{cfg.SEED},project={cfg.DESCRIPTION.PROJECT_NAME}"

        cfg = get_cfg()
        cfg.DESCRIPTION.BRIEF = 'f"{SEED},project={DESCRIPTION.PROJECT_NAME}-today"'
        cfg.format_fields()
        assert cfg.DESCRIPTION.BRIEF == f"{cfg.SEED},project={cfg.DESCRIPTION.PROJECT_NAME}-today"

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

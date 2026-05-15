import os
import tempfile
import unittest
from unittest.mock import patch

from app.services.youtube_service import YoutubeService


class YoutubeServiceTest(unittest.TestCase):
    def setUp(self):
        self.service = YoutubeService()

    def test_normalize_youtube_url_accepts_common_hosts(self):
        self.assertEqual(
            "https://youtu.be/example",
            self.service._normalize_youtube_url("youtu.be/example"),
        )
        self.assertEqual(
            "https://www.youtube.com/watch?v=example",
            self.service._normalize_youtube_url("https://www.youtube.com/watch?v=example"),
        )

    def test_normalize_youtube_url_rejects_other_hosts(self):
        with self.assertRaises(ValueError):
            self.service._normalize_youtube_url("https://example.com/watch?v=example")

    def test_sanitize_filename_stem_removes_path_separators(self):
        self.assertEqual(
            "bad_name_ test",
            self.service._sanitize_filename_stem("../bad/name? test"),
        )

    def test_resolution_height_parses_labels(self):
        self.assertEqual(720, self.service._resolution_height("720p"))
        self.assertEqual(1080, self.service._resolution_height("1080p60"))
        self.assertIsNone(self.service._resolution_height("best"))

    def test_ffmpeg_location_falls_back_to_python_environment(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            fake_python = os.path.join(temp_dir, "python")
            fake_ffmpeg = os.path.join(temp_dir, "ffmpeg")
            with open(fake_ffmpeg, "w", encoding="utf-8"):
                pass

            with patch("app.services.youtube_service.shutil.which", return_value=None), patch(
                "app.services.youtube_service.sys.executable",
                fake_python,
            ):
                self.assertEqual(fake_ffmpeg, self.service._ffmpeg_location())


if __name__ == "__main__":
    unittest.main()

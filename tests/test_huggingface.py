import unittest
from unittest.mock import MagicMock, patch

from server import huggingface


class HuggingFaceTests(unittest.TestCase):
    def test_search_returns_gguf_model_summary(self):
        payload = [
            {
                "id": "owner/model-GGUF",
                "downloads": 1234,
                "likes": 12,
                "tags": ["gguf", "license:apache-2.0"],
                "siblings": [
                    {"rfilename": "model-q4.gguf"},
                    {"rfilename": "README.md"},
                ],
            }
        ]
        with patch.object(huggingface, "_request_json", return_value=payload):
            result = huggingface.search_models("model")
        self.assertEqual(result[0]["id"], "owner/model-GGUF")
        self.assertEqual(result[0]["author"], "owner")
        self.assertEqual(result[0]["avatar_url"], "https://huggingface.co/avatars/owner.svg")
        self.assertEqual(result[0]["license"], "apache-2.0")
        self.assertEqual(result[0]["gguf_count"], 1)

    def test_search_sends_sort_and_direction(self):
        with patch.object(huggingface, "_request_json", return_value=[]) as request:
            huggingface.search_models("model", sort="likes", direction=1)
        self.assertEqual(request.call_args.kwargs["params"]["sort"], "likes")
        self.assertEqual(request.call_args.kwargs["params"]["direction"], 1)

    def test_model_files_returns_only_gguf_files(self):
        payload = {
            "siblings": [
                {"rfilename": "model-q4.gguf", "size": 100},
                {"rfilename": "README.md", "size": 20},
            ]
        }
        with patch.object(huggingface, "_request_json", return_value=payload):
            result = huggingface.model_files("owner/model")
        self.assertEqual(result["files"], [{"name": "model-q4.gguf", "size": 100}])

    def test_download_rejects_unsafe_file_paths(self):
        with self.assertRaisesRegex(ValueError, "Only GGUF"):
            huggingface.start_download("owner/model", ["../model.gguf"])

    def test_download_starts_a_background_job(self):
        info = {
            "files": [{"name": "model-q4.gguf", "size": 100}],
            "id": "owner/model",
        }
        thread = MagicMock()
        with (
            patch.object(huggingface, "model_files", return_value=info),
            patch.object(huggingface.threading, "Thread", return_value=thread),
        ):
            job = huggingface.start_download("owner/model", ["model-q4.gguf"])
        self.assertEqual(job["status"], "downloading")
        self.assertEqual(job["total_bytes"], 100)
        thread.start.assert_called_once()


if __name__ == "__main__":
    unittest.main()

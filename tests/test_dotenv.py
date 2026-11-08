"""harness.config.load_dotenv 테스트."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from harness.config import load_dotenv


class DotenvTest(unittest.TestCase):
    def _run(self, text: str, env: dict[str, str]) -> tuple[list[str], dict[str, str]]:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / ".env"
            p.write_text(text, encoding="utf-8")
            with mock.patch.dict(os.environ, env, clear=True):
                applied = load_dotenv(p)
                return applied, dict(os.environ)

    def test_parses_quotes_comments_and_export(self):
        applied, env = self._run("# comment\nANTHROPIC_API_KEY='sk-test-123'\nexport LLM_MODEL=\"claude-sonnet-5-5\"\n\nbad name=1\n", {})
        self.assertEqual(applied, ["ANTHROPIC_API_KEY", "LLM_MODEL"])
        self.assertEqual(env["ANTHROPIC_API_KEY"], "sk-test-123")
        self.assertEqual(env["LLM_MODEL"], "claude-sonnet-5-5")
        self.assertNotIn("bad name", env)

    def test_does_not_override_existing(self):
        applied, env = self._run("ANTHROPIC_API_KEY=from-file\n", {"ANTHROPIC_API_KEY": "from-shell"})
        self.assertEqual(applied, [])
        self.assertEqual(env["ANTHROPIC_API_KEY"], "from-shell")

    def test_missing_file_is_noop(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(load_dotenv(Path("/nonexistent/.env")), [])


if __name__ == "__main__":
    unittest.main()

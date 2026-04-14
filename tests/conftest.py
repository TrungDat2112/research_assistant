"""Shared test fixtures for CommitLens."""

from __future__ import annotations

import pytest

SAMPLE_DIFF_TEXT = """\
diff --git a/hello.py b/hello.py
--- a/hello.py
+++ b/hello.py
@@ -1,3 +1,5 @@
 def greet(name):
-    print("hello")
+    if not name:
+        raise ValueError("name required")
+    print(f"hello {name}")
"""


@pytest.fixture()
def sample_diff_text() -> str:
    return SAMPLE_DIFF_TEXT

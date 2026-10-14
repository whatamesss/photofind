"""Lightweight tests for PhotoFind's quality scoring, duplicate ranking, folder
overlap analysis, and settings persistence.

These avoid constructing the CLIP model by allocating a bare PhotoSearch object.
Run with:  python -m unittest discover -s tests
"""

import importlib.util
import json
import os
import tempfile
import threading
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("photofind", _ROOT / "photofind.py")
pf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pf)


def bare_searcher():
    s = object.__new__(pf.PhotoSearch)
    s.quality_settings = dict(pf.DEFAULT_QUALITY_SETTINGS)
    s.search_settings = dict(pf.DEFAULT_SEARCH_SETTINGS)
    s.image_paths = []
    s.image_metadata = {}
    s.indexed_set = set()
    s.image_embeddings = None
    s._dir_counts = None
    s._lock = threading.Lock()
    return s


class QualityScorersTest(unittest.TestCase):
    def test_darkness(self):
        s = bare_searcher()
        self.assertAlmostEqual(s._darkness({"lum_p95": 0.0}), 1.0)
        self.assertEqual(s._darkness({"lum_p95": 200.0}), 0.0)
        self.assertTrue(0.0 < s._darkness({"lum_p95": 15.0}) < 1.0)

    def test_blur(self):
        s = bare_searcher()
        self.assertAlmostEqual(s._blur({"lap_p95": 0.0}), 1.0)
        self.assertEqual(s._blur({"lap_p95": 50.0}), 0.0)

    def test_overexposure(self):
        s = bare_searcher()
        self.assertAlmostEqual(s._overexposure({"lum_p95": 255.0}), 1.0)
        self.assertAlmostEqual(s._overexposure({"mean_lum": 255.0}), 1.0)
        self.assertEqual(s._overexposure({"lum_p95": 100.0, "mean_lum": 100.0}), 0.0)

    def test_thresholds_are_respected(self):
        s = bare_searcher()
        meta = {"lum_p95": 20.0}
        self.assertGreater(s._darkness(meta), 0.0)
        s.quality_settings["dark_lum_p95"] = 5.0
        self.assertEqual(s._darkness(meta), 0.0)

    def test_count_and_get_problem_photos(self):
        s = bare_searcher()
        tmp = tempfile.mkdtemp()
        dark = os.path.join(tmp, "dark.jpg")
        bright = os.path.join(tmp, "bright.jpg")
        for p in (dark, bright):
            with open(p, "w") as f:
                f.write("x")
        s.image_paths = [dark, bright]
        s.indexed_set = set(s.image_paths)
        s.image_metadata = {
            dark: {"lum_p95": 0.0, "lap_p95": 50.0, "mean_lum": 0.0},
            bright: {"lum_p95": 200.0, "lap_p95": 50.0, "mean_lum": 150.0},
        }
        counts = s.count_problem_photos()
        self.assertEqual(counts["dark"], 1)
        hits = s.get_problem_photos("dark")
        self.assertEqual([h["file"] for h in hits], [dark])
        self.assertEqual(hits[0]["kind"], "dark")


class DuplicateRankingTest(unittest.TestCase):
    def test_prefers_library_over_junk_and_bigger_dir(self):
        s = bare_searcher()
        keep = "/photos/library/2020/pic.jpg"
        other = "/photos/library/2020/pic2.jpg"
        junk = "/photos/downloads/pic.jpg"
        s.image_paths = [keep, other, junk]
        ranked = s.rank_duplicate_group([junk, keep])
        self.assertEqual(ranked[0], keep)

    def test_shallower_path_wins_when_tied(self):
        s = bare_searcher()
        shallow = "/photos/pic.jpg"
        deep = "/photos/a/b/c/pic.jpg"
        s.image_paths = [shallow, deep]
        self.assertEqual(s.rank_duplicate_group([deep, shallow])[0], shallow)

    def test_unique_dest_path(self):
        tmp = tempfile.mkdtemp()
        with open(os.path.join(tmp, "a.jpg"), "w") as f:
            f.write("x")
        self.assertEqual(pf.PhotoSearch._unique_dest_path(tmp, "b.jpg"), os.path.join(tmp, "b.jpg"))
        self.assertEqual(pf.PhotoSearch._unique_dest_path(tmp, "a.jpg"), os.path.join(tmp, "a_1.jpg"))


class FolderOverlapTest(unittest.TestCase):
    def test_mirror_and_subset(self):
        s = bare_searcher()
        a1, a2 = "/lib/A/1.jpg", "/lib/A/2.jpg"
        b1, b2 = "/lib/B/1.jpg", "/lib/B/2.jpg"
        s.image_paths = [a1, a2, b1, b2]
        s._dir_counts = None
        groups = [[a1, b1], [a2, b2]]
        stats = s.compute_folder_duplicate_stats(groups)
        self.assertEqual(len(stats), 1)
        self.assertEqual(stats[0]["kind"], "mirror")
        self.assertAlmostEqual(stats[0]["cov_a"], 1.0)
        self.assertAlmostEqual(stats[0]["cov_b"], 1.0)

    def test_plan_merge_splits_unique_and_redundant(self):
        s = bare_searcher()
        a1, a2, b1 = "/lib/A/1.jpg", "/lib/A/2.jpg", "/lib/B/1.jpg"
        s.image_paths = [a1, a2, b1]
        groups = [[a1, b1]]
        unique, redundant = s.plan_merge("/lib/A", "/lib/B", groups)
        self.assertEqual(unique, [a2])
        self.assertEqual(redundant, [a1])


class SettingsTest(unittest.TestCase):
    def test_load_merges_over_defaults(self):
        s = bare_searcher()
        tmp = tempfile.mkdtemp()
        s.settings_file = Path(tmp) / "settings.json"
        s.search_settings = dict(pf.DEFAULT_SEARCH_SETTINGS)
        s.quality_settings = dict(pf.DEFAULT_QUALITY_SETTINGS)
        with open(s.settings_file, "w") as f:
            json.dump({"search": {"top_k": 12}, "quality": {"dark_lum_p95": 3.0}}, f)
        s._load_settings()
        self.assertEqual(s.search_settings["top_k"], 12)
        self.assertEqual(s.quality_settings["dark_lum_p95"], 3.0)
        self.assertEqual(s.search_settings["floor_min"], pf.DEFAULT_SEARCH_SETTINGS["floor_min"])


if __name__ == "__main__":
    unittest.main()

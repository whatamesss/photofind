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
from unittest import mock

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
        # Featureless frames (low edge / blank) are never "blurry".
        self.assertEqual(s._blur({"lap_var": 0.0, "std_dev": 0.0}), 0.0)
        self.assertEqual(s._blur({"lap_var": 0.0, "edge_score": 0.5, "std_dev": 20.0}), 0.0)
        # A photo with real content: low Laplacian variance = blurry.
        self.assertGreater(s._blur({"lap_var": 0.0, "edge_score": 20.0, "std_dev": 40.0}), 0.0)
        self.assertEqual(s._blur({"lap_var": 5000.0, "edge_score": 20.0, "std_dev": 40.0}), 0.0)

    def test_blank(self):
        s = bare_searcher()
        self.assertAlmostEqual(s._blankness({"std_dev": 0.0}), 1.0)
        self.assertEqual(s._blankness({"std_dev": 50.0}), 0.0)
        self.assertTrue(s._is_blank({"std_dev": 0.0}))
        self.assertFalse(s._is_blank({"std_dev": 40.0}))

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
        self.assertEqual([h["file"] for h in hits], [dark, bright])
        self.assertEqual(hits[0]["kind"], "dark")
        self.assertFalse(hits[0]["below_threshold"])
        self.assertTrue(hits[1]["below_threshold"])

    def test_problem_photos_never_empty_when_photos_exist(self):
        # Even with no photo past the cutoff, the worst-ranked photo is returned
        # so the filters stay useful instead of appearing broken.
        s = bare_searcher()
        tmp = tempfile.mkdtemp()
        paths = []
        for i in range(3):
            p = os.path.join(tmp, f"p{i}.jpg")
            with open(p, "w") as f:
                f.write("x")
            paths.append(p)
        s.image_paths = list(paths)
        s.indexed_set = set(paths)
        s.image_metadata = {
            paths[0]: {"lum_p95": 150.0, "lap_var": 100.0, "edge_score": 20.0, "std_dev": 40.0, "mean_lum": 140.0},
            paths[1]: {"lum_p95": 160.0, "lap_var": 30.0, "edge_score": 20.0, "std_dev": 40.0, "mean_lum": 135.0},
            paths[2]: {"lum_p95": 170.0, "lap_var": 50.0, "edge_score": 20.0, "std_dev": 40.0, "mean_lum": 150.0},
        }
        self.assertEqual(s.count_problem_photos()["blurry"], 0)
        hits = s.get_problem_photos("blurry")
        self.assertEqual(len(hits), 3)
        self.assertTrue(all(h["below_threshold"] for h in hits))
        # lowest lap_var is worst / first
        self.assertEqual(hits[0]["file"], paths[1])

    def test_blank_frames_not_counted_as_blurry(self):
        s = bare_searcher()
        tmp = tempfile.mkdtemp()
        blank = os.path.join(tmp, "blank.jpg")
        photo = os.path.join(tmp, "photo.jpg")
        for p in (blank, photo):
            with open(p, "w") as f:
                f.write("x")
        s.image_paths = [blank, photo]
        s.indexed_set = set(s.image_paths)
        s.image_metadata = {
            blank: {"lum_p95": 128.0, "lap_var": 0.0, "edge_score": 0.0, "std_dev": 0.0, "mean_lum": 128.0},
            photo: {"lum_p95": 128.0, "lap_var": 500.0, "edge_score": 30.0, "std_dev": 45.0, "mean_lum": 128.0},
        }
        # The solid frame is blank, never blurry.
        self.assertEqual(s._blur(s.image_metadata[blank]), 0.0)
        self.assertEqual(s.count_problem_photos()["blurry"], 0)
        blank_hits = s.get_problem_photos("blank")
        self.assertEqual(blank_hits[0]["file"], blank)
        self.assertNotIn(blank, [h["file"] for h in s.get_problem_photos("blurry", 0, 1) if h["score"] > 0])

    def test_problem_severity_ordering(self):
        s = bare_searcher()
        self.assertGreater(s._problem_severity("dark", {"lum_p95": 0.0}),
                           s._problem_severity("dark", {"lum_p95": 20.0}))
        self.assertGreater(s._problem_severity("overexposed", {"lum_p95": 255.0}),
                           s._problem_severity("overexposed", {"lum_p95": 240.0}))
        self.assertIsNone(s._problem_severity("blurry", {"lum_p95": 50.0}))


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

    def test_viewer_setting_roundtrip(self):
        s = bare_searcher()
        s.viewer_settings = dict(pf.DEFAULT_VIEWER_SETTINGS)
        tmp = tempfile.mkdtemp()
        s.settings_file = Path(tmp) / "settings.json"
        s.save_settings(viewer={"image_viewer": "gwenview"})
        with open(s.settings_file) as f:
            data = json.load(f)
        self.assertEqual(data["viewer"]["image_viewer"], "gwenview")
        # Reload into a fresh object and confirm it is restored.
        s2 = bare_searcher()
        s2.viewer_settings = dict(pf.DEFAULT_VIEWER_SETTINGS)
        s2.settings_file = s.settings_file
        s2._load_settings()
        self.assertEqual(s2.viewer_settings["image_viewer"], "gwenview")


class ViewerDetectionTest(unittest.TestCase):
    """detect_image_viewers() derives its list from .desktop entries, so we
    point it at a temporary applications directory rather than the real system.
    """

    def _desktop(self, appdir, filename, *, name, mimes, exec_line, hidden=False):
        body = [f"[Desktop Entry]", "Type=Application", f"Name={name}",
                f"Exec={exec_line}", f"MimeType={mimes}"]
        if hidden:
            body.append("Hidden=true")
        with open(os.path.join(appdir, filename), "w") as f:
            f.write("\n".join(body) + "\n")

    def test_detects_image_apps_and_skips_others(self):
        tmp = tempfile.mkdtemp()
        home = os.path.join(tmp, "home"); ad = os.path.join(home, "applications")
        data = os.path.join(tmp, "data"); ad2 = os.path.join(data, "applications")
        os.makedirs(ad); os.makedirs(ad2)
        # Resolvable executable that exists on all Linux systems.
        self.assertEqual(pf._find_executable("/bin/true"), True)
        self._desktop(ad, "imgy.desktop", name="ImgY", mimes="image/jpeg;image/png;",
                      exec_line="/bin/true %U")
        # A non-image app must not appear.
        self._desktop(ad, "text.desktop", name="TextY", mimes="text/plain;",
                      exec_line="/bin/true %U")
        # A hidden image app must not appear.
        self._desktop(ad2, "hidden.desktop", name="HiddenY", mimes="image/png;",
                      exec_line="/bin/true %U", hidden=True)
        # env-wrapped Exec should resolve to the real executable.
        self._desktop(ad2, "envimg.desktop", name="EnvY", mimes="image/png;",
                      exec_line="env GTK_THEME=x /bin/echo %U")

        env = {"XDG_DATA_HOME": home, "XDG_DATA_DIRS": data}
        with mock.patch.dict(os.environ, env, clear=False):
            viewers = pf.detect_image_viewers()

        self.assertIsInstance(viewers, list)
        for item in viewers:
            self.assertIsInstance(item, tuple)
            self.assertEqual(len(item), 2)
            self.assertIsInstance(item[0], str); self.assertIsInstance(item[1], str)
        labels = {label for _cmd, label in viewers}
        self.assertIn("ImgY", labels)
        self.assertIn("EnvY", labels)
        self.assertNotIn("TextY", labels)
        self.assertNotIn("HiddenY", labels)
        # Every returned command must be launchable.
        for cmd, _label in viewers:
            self.assertTrue(pf._find_executable(cmd), f"{cmd!r} not resolvable")


class ViewerLaunchTest(unittest.TestCase):
    """The launcher special-cases Geeqie (fullscreen start, single-instance)."""

    def _launch(self, command, path="/tmp/pic.jpg"):
        captured = {}

        class FakePopen:
            def __init__(self, argv, **kwargs): captured["argv"] = argv

        with mock.patch.object(pf.subprocess, "Popen", FakePopen):
            ok = pf.PhotoOrganizerWindow._launch_image_viewer(None, command, path)
        return ok, captured.get("argv")

    def test_geeqie_gets_fullscreen_flag(self):
        ok, argv = self._launch("geeqie")
        self.assertTrue(ok)
        self.assertEqual(argv, ["geeqie", "--fullscreen", "/tmp/pic.jpg"])

    def test_geeqie_matched_case_and_path_insensitively(self):
        ok, argv = self._launch("/usr/bin/geeqie")
        self.assertEqual(argv, ["/usr/bin/geeqie", "--fullscreen", "/tmp/pic.jpg"])

    def test_other_viewer_unmodified(self):
        ok, argv = self._launch("feh")
        self.assertEqual(argv, ["feh", "/tmp/pic.jpg"])

    def test_viewer_with_args_keeps_args_then_appends_path(self):
        ok, argv = self._launch("flatpak run org.example.View")
        self.assertEqual(argv, ["flatpak", "run", "org.example.View", "/tmp/pic.jpg"])


if __name__ == "__main__":
    unittest.main()

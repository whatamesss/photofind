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

from PIL import Image

_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("photofind", _ROOT / "photofind.py")
pf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pf)


def bare_searcher():
    s = object.__new__(pf.PhotoSearch)
    s.quality_settings = dict(pf.DEFAULT_QUALITY_SETTINGS)
    s.search_settings = dict(pf.DEFAULT_SEARCH_SETTINGS)
    s.viewer_settings = dict(pf.DEFAULT_VIEWER_SETTINGS)
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

    def test_default_viewer_is_gwenview(self):
        self.assertEqual(pf.DEFAULT_VIEWER_SETTINGS["image_viewer"], "gwenview")

    def test_legacy_system_viewer_migrates_to_gwenview(self):
        s = bare_searcher()
        tmp = tempfile.mkdtemp()
        s.settings_file = Path(tmp) / "settings.json"
        with open(s.settings_file, "w") as f:
            json.dump({"viewer": {"image_viewer": "system"}}, f)
        s._load_settings()
        self.assertEqual(s.viewer_settings["image_viewer"], "gwenview")

    def test_explicit_viewer_not_migrated(self):
        s = bare_searcher()
        tmp = tempfile.mkdtemp()
        s.settings_file = Path(tmp) / "settings.json"
        with open(s.settings_file, "w") as f:
            json.dump({"viewer": {"image_viewer": "geeqie"}}, f)
        s._load_settings()
        self.assertEqual(s.viewer_settings["image_viewer"], "geeqie")


class ClipboardCopyTest(unittest.TestCase):
    """Right-click "Copy Image to Clipboard" copies pixels, not a file URL."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def _png(self, name, size=(4, 3), color=(10, 20, 30, 128)):
        path = os.path.join(self.tmp, name)
        Image.new("RGBA", size, color).save(path)
        return path

    def test_rgba_pixels_and_alpha_preserved(self):
        qimg = pf.load_image_for_clipboard(self._png("rgba.png"))
        self.assertFalse(qimg.isNull())
        self.assertEqual((qimg.width(), qimg.height()), (4, 3))
        self.assertEqual(qimg.pixelColor(1, 1).getRgb(), (10, 20, 30, 128))

    def test_palette_and_greyscale_are_converted(self):
        pal = os.path.join(self.tmp, "pal.png")
        Image.new("P", (2, 2), 5).save(pal)
        grey = os.path.join(self.tmp, "grey.png")
        Image.new("L", (3, 1), 99).save(grey)
        for path in (pal, grey):
            qimg = pf.load_image_for_clipboard(path)
            self.assertFalse(qimg.isNull(), path)
            self.assertEqual(qimg.format(), pf.QImage.Format.Format_RGBA8888)

    def test_exif_orientation_is_applied(self):
        # A landscape file tagged "rotate 90" is displayed portrait.
        path = os.path.join(self.tmp, "rotated.jpg")
        img = Image.new("RGB", (8, 4), (200, 100, 50))
        exif = Image.Exif(); exif[0x0112] = 6
        img.save(path, exif=exif)
        qimg = pf.load_image_for_clipboard(path)
        self.assertEqual((qimg.width(), qimg.height()), (4, 8))

    def test_unreadable_file_returns_message(self):
        err = pf.copy_image_to_clipboard(os.path.join(self.tmp, "missing.jpg"))
        self.assertIsInstance(err, str)
        self.assertIn("Could not read image", err)


class ViewerDetectionTest(unittest.TestCase):
    """detect_image_viewers() derives its list from .desktop entries, so we
    point it at a temporary applications directory plus a temp bin dir of fake
    executables, rather than touching the real system.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.home = os.path.join(self.tmp, "home"); self.appdir = os.path.join(self.home, "applications")
        self.data = os.path.join(self.tmp, "data"); self.appdir2 = os.path.join(self.data, "applications")
        self.bindir = os.path.join(self.tmp, "bin")
        for d in (self.appdir, self.appdir2, self.bindir):
            os.makedirs(d, exist_ok=True)
        # Fake executables for both allow-listed photo apps and a non-photo app.
        for exe in ("gwenview", "geeqie", "gimp", "eog", "ristretto",
                    "firefox", "libreoffice"):
            p = os.path.join(self.bindir, exe)
            with open(p, "w") as f:
                f.write("#!/bin/sh\n")
            os.chmod(p, 0o755)

    def _env(self):
        return {"XDG_DATA_HOME": self.home, "XDG_DATA_DIRS": self.data,
                "PATH": self.bindir + os.pathsep + os.environ.get("PATH", "")}

    def _desktop(self, appdir, filename, *, name, mimes, exec_line, hidden=False):
        body = [f"[Desktop Entry]", "Type=Application", f"Name={name}",
                f"Exec={exec_line}", f"MimeType={mimes}"]
        if hidden:
            body.append("Hidden=true")
        with open(os.path.join(appdir, filename), "w") as f:
            f.write("\n".join(body) + "\n")

    def test_detects_photo_apps_only(self):
        self._desktop(self.appdir, "gwenview.desktop", name="Gwenview",
                      mimes="image/jpeg;image/png;", exec_line="gwenview %U")
        self._desktop(self.appdir, "geeqie.desktop", name="Geeqie",
                      mimes="image/*", exec_line="geeqie %U")
        self._desktop(self.appdir, "gimp.desktop", name="GIMP",
                      mimes="image/png;image/x-xcf;", exec_line="gimp %U")
        self._desktop(self.appdir, "eog.desktop", name="Image Viewer",
                      mimes="image/png;", exec_line="eog %U")
        # A browser and an office app that merely register for image/* must be
        # excluded as inappropriate photo-viewer choices.
        self._desktop(self.appdir, "browser.desktop", name="Firefox",
                      mimes="image/jpeg;", exec_line="firefox %U")
        self._desktop(self.appdir, "office.desktop", name="LibreOffice Draw",
                      mimes="image/png;", exec_line="libreoffice --draw %U")
        # A non-image app and a hidden image app must not appear.
        self._desktop(self.appdir2, "text.desktop", name="TextY",
                      mimes="text/plain;", exec_line="gwenview %U")
        self._desktop(self.appdir2, "hidden.desktop", name="HiddenY",
                      mimes="image/png;", exec_line="geeqie %U", hidden=True)
        # env-wrapped Exec should resolve to the real executable.
        self._desktop(self.appdir2, "envimg.desktop", name="EnvY",
                      mimes="image/png;", exec_line="env GTK_THEME=x ristretto %U")

        with mock.patch.dict(os.environ, self._env(), clear=False):
            viewers = pf.detect_image_viewers()
            resolvable = {cmd: pf._find_executable(cmd) for cmd, _ in viewers}

        self.assertIsInstance(viewers, list)
        for item in viewers:
            self.assertIsInstance(item, tuple)
            self.assertEqual(len(item), 2)
            self.assertIsInstance(item[0], str); self.assertIsInstance(item[1], str)

        labels = [label for _cmd, label in viewers]
        self.assertIn("Gwenview", labels)
        self.assertIn("Geeqie", labels)
        self.assertIn("GIMP", labels)
        self.assertIn("Image Viewer", labels)   # eog
        self.assertIn("EnvY", labels)           # env-wrapped eog
        # Excluded: non-photo apps, non-image apps, hidden apps.
        self.assertNotIn("Firefox", labels)
        self.assertNotIn("LibreOffice Draw", labels)
        self.assertNotIn("TextY", labels)
        self.assertNotIn("HiddenY", labels)

        # Preferred order: Gwenview first, then Geeqie, then GIMP.
        self.assertEqual(viewers[0][1], "Gwenview")
        self.assertEqual([label for _cmd, label in viewers[:3]],
                         ["Gwenview", "Geeqie", "GIMP"])

        # Every returned command must be launchable.
        for cmd in resolvable:
            self.assertTrue(resolvable[cmd], f"{cmd!r} not resolvable")

    def test_photo_viewer_name_helpers(self):
        self.assertEqual(pf._photo_viewer_name("gwenview %U"), "gwenview")
        self.assertEqual(pf._photo_viewer_name("/usr/bin/gimp-2.10"), "gimp-2.10")
        self.assertEqual(pf._photo_viewer_name("flatpak run org.gnome.Loupe"), "loupe")
        self.assertIsNone(pf._photo_viewer_name("firefox %U"))
        self.assertIsNone(pf._photo_viewer_name("libreoffice --draw"))


class ViewerLaunchTest(unittest.TestCase):
    """The launcher special-cases Geeqie (fresh full-screen instance)."""

    def _launch(self, command, path="/tmp/pic.jpg"):
        captured = {}

        class FakePopen:
            def __init__(self, argv, **kwargs): captured.update(kwargs, argv=argv)
            def poll(self): return 0

        with mock.patch.object(pf.subprocess, "Popen", FakePopen):
            ok = pf.PhotoOrganizerWindow._launch_image_viewer(None, command, path)
        return ok, captured.get("argv"), captured.get("env")

    def test_geeqie_fullscreen_and_new_instance(self):
        ok, argv, env = self._launch("geeqie")
        self.assertTrue(ok)
        self.assertEqual(argv, ["geeqie", "--fullscreen", "/tmp/pic.jpg"])
        self.assertEqual(env.get("GQ_NEW_INSTANCE"), "yes")

    def test_geeqie_matched_case_and_path_insensitively(self):
        ok, argv, env = self._launch("/usr/bin/geeqie")
        self.assertEqual(argv, ["/usr/bin/geeqie", "--fullscreen", "/tmp/pic.jpg"])
        self.assertEqual(env.get("GQ_NEW_INSTANCE"), "yes")

    def test_other_viewer_unmodified(self):
        ok, argv, env = self._launch("feh")
        self.assertEqual(argv, ["feh", "/tmp/pic.jpg"])
        self.assertIsNone(env)

    def test_viewer_with_args_keeps_args_then_appends_path(self):
        ok, argv, env = self._launch("flatpak run org.example.View")
        self.assertEqual(argv, ["flatpak", "run", "org.example.View", "/tmp/pic.jpg"])
        self.assertIsNone(env)

    def test_reap_drops_finished_keeps_running(self):
        class Proc:
            def __init__(self, rc): self._rc = rc
            def poll(self): return self._rc

        alive, dead = Proc(None), Proc(0)
        pf._VIEWER_PROCESSES[:] = [alive, dead]
        pf.reap_viewer_processes()
        self.assertEqual(pf._VIEWER_PROCESSES, [alive])
        pf._VIEWER_PROCESSES[:] = []


if __name__ == "__main__":
    unittest.main()

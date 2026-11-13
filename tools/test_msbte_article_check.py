#!/usr/bin/env python3
"""Regression checks for portable, fail-closed MS-BTE notebook replay."""
from __future__ import annotations

import base64
import contextlib
import copy
import io
import json
from pathlib import Path
import unittest
from unittest import mock

import build_msbte_notebooks
import check_msbte_article
from check_msbte_article import notebook_outputs_match


def stream(text, name="stdout"):
    return [{"output_type": "stream", "name": name, "text": text}]


class NotebookOutputChecks(unittest.TestCase):
    def test_builder_renders_figures_by_default(self):
        rendered_png = b"rendered PNG regression sentinel"
        with mock.patch("build_msbte_notebooks.subprocess.check_output", return_value=rendered_png) as renderer:
            cells = build_msbte_notebooks.notebook_cells("en")
        attachments = {name: formats for cell in cells for name, formats in cell.get("attachments", {}).items()}
        self.assertEqual(len(attachments), 7)
        self.assertEqual(renderer.call_count, 7)
        for call in renderer.call_args_list:
            self.assertEqual(call.args[0][:3], ["rsvg-convert", "--format", "png"])
            self.assertTrue(call.args[0][3].endswith(".svg"))
        for formats in attachments.values():
            self.assertEqual(base64.b64decode(formats["image/png"], validate=True), rendered_png)

    def test_actual_checker_replays_both_notebooks_without_external_renderer(self):
        output = io.StringIO()
        with mock.patch("build_msbte_notebooks.subprocess.check_output",
                        side_effect=FileNotFoundError("rsvg-convert unavailable in regression")) as renderer:
            with contextlib.redirect_stdout(output):
                check_msbte_article.main()
        renderer.assert_not_called()
        result = json.loads(output.getvalue())
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["executed_notebooks"], 2)
        self.assertEqual(result["replayed_notebook_cells"], 24)
        self.assertEqual(result["byte_exact_embedded_figures"], 14)

    def test_recorded_blas_roundoff(self):
        stored = stream("occupation range: 0.22023121387283215 0.8797687861271675\n"
                        "face-flux variation: 7.327471962526033e-15\n")
        replayed = stream("occupation range: 0.22023121387283215 0.8797687861271674\n"
                          "face-flux variation: 7.66053886991358e-15\n")
        self.assertTrue(notebook_outputs_match(stored, replayed))
        self.assertTrue(notebook_outputs_match(replayed, stored))

    def test_recorded_current_roundoff(self):
        self.assertTrue(notebook_outputs_match(
            stream("I_e, ballistic / elastic model (uA): 7.397701845724329 2.1874518520064594\n"),
            stream("I_e, ballistic / elastic model (uA): 7.397701845724329 2.187451852006462\n")))

    def test_scientific_notation_and_near_zero_roundoff(self):
        self.assertTrue(notebook_outputs_match(stream("value: +1.25e-02\n"), stream("value: 0.01250000000000001\n")))
        self.assertTrue(notebook_outputs_match(stream("residual: 0.0\n"), stream("residual: -7.7e-15\n")))

    def test_material_numeric_changes_fail(self):
        for old, new in [("0.8797687861271675", "0.89"),
                         ("2.1874518520064594", "2.187451853"),
                         ("7.3e-15", "1e-7"), ("1.0e9", "1.0e9 1")]:
            with self.subTest(old=old, new=new):
                self.assertFalse(notebook_outputs_match(stream("value: " + old), stream("value: " + new)))

    def test_integer_counts_remain_exact(self):
        self.assertFalse(notebook_outputs_match(stream("cases: 12\n"), stream("cases: 13\n")))
        # Relative float tolerance would incorrectly accept this changed count.
        self.assertFalse(notebook_outputs_match(stream("cases: 1000000000000000\n"),
                                                stream("cases: 1000000000000001\n")))

    def test_text_skeleton_is_exact(self):
        for changed in ["current = 2.0\n", "Current: 2.0\n", "current: 2.0",
                        "current: 2.0\nnew text\n", "current: [2.0]\n"]:
            with self.subTest(changed=changed):
                self.assertFalse(notebook_outputs_match(stream("current: 2.0\n"), stream(changed)))

    def test_numeric_token_count_and_positions_are_exact(self):
        self.assertFalse(notebook_outputs_match(stream("range: 0.2 0.8\n"), stream("range: 0.2\n")))
        self.assertFalse(notebook_outputs_match(stream("1 <number>\n"), stream("<number> 1\n")))

    def test_stream_count_and_metadata_are_exact(self):
        self.assertTrue(notebook_outputs_match([], []))
        self.assertFalse(notebook_outputs_match([], stream("")))
        self.assertFalse(notebook_outputs_match(stream("ok\n"), stream("ok\n") * 2))
        self.assertFalse(notebook_outputs_match(stream("ok\n"), stream("ok\n", name="stderr")))
        extra = stream("ok\n")
        extra[0]["metadata"] = {}
        self.assertFalse(notebook_outputs_match(stream("ok\n"), extra))

    def test_nonfinite_tokens_fail_even_when_identical(self):
        for value in ["nan", "NaN", "+nan", "-NAN", "inf", "+Inf", "-inf", "Infinity", "-infinity"]:
            with self.subTest(value=value):
                output = stream("value: " + value + "\n")
                self.assertFalse(notebook_outputs_match(output, copy.deepcopy(output)))

    def test_float_overflow_fails_even_when_identical(self):
        for value in ["1e400", "-1E9999"]:
            with self.subTest(value=value):
                self.assertFalse(notebook_outputs_match(stream(value), stream(value)))

    def test_malformed_or_nonstream_outputs_fail(self):
        malformed = [None, {}, "ok", [None], [{"output_type": "error", "name": "stdout", "text": "ok"}],
                     [{"output_type": "stream", "name": "stdout"}], stream(["ok\n"])]
        for output in malformed:
            with self.subTest(output=output):
                self.assertFalse(notebook_outputs_match(output, stream("ok")))
                self.assertFalse(notebook_outputs_match(stream("ok"), output))

    def test_output_order_is_preserved(self):
        first, second = stream("first: 1.0\n"), stream("second: 2.0\n")
        self.assertFalse(notebook_outputs_match(first + second, second + first))

    def test_nonnumeric_text_is_not_mistaken_for_nonfinite_values(self):
        output = stream("finite grid, infinite well, nanoparticle\n")
        before = copy.deepcopy(output)
        self.assertTrue(notebook_outputs_match(output, copy.deepcopy(output)))
        self.assertEqual(output, before)

    def test_checked_in_notebook_streams_are_valid_and_extra_text_fails(self):
        root = Path(__file__).resolve().parents[1]
        for language in ("en", "ko"):
            path = root / f"assets/notebooks/multi-subband-bte-{language}.ipynb"
            notebook = json.loads(path.read_text(encoding="utf-8"))
            for index, cell in enumerate(notebook["cells"]):
                if cell["cell_type"] != "code":
                    continue
                with self.subTest(language=language, cell=index):
                    stored = cell["outputs"]
                    self.assertTrue(notebook_outputs_match(stored, copy.deepcopy(stored)))
                    if stored:
                        changed = copy.deepcopy(stored)
                        changed[0]["text"] += "unexpected output\n"
                        self.assertFalse(notebook_outputs_match(stored, changed))


if __name__ == "__main__":
    unittest.main()

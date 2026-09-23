"""Exercise diagnostic hooks without loading CUDA, models, or notebook servers."""
import ast
import pathlib
import types
import unittest


class TelemetryIsolationTests(unittest.TestCase):
    def test_all_variants_preserve_work_when_telemetry_fails(self):
        for filename in ("h3_studio.py", "h3_studio_3.py", "h3_studio_l40s.py"):
            with self.subTest(filename=filename):
                source = ast.parse(pathlib.Path(__file__).with_name(filename).read_text(encoding="utf-8-sig"))
                names = {"_ml_observe", "_set_job_stage", "_ml_report_api_failure"}
                functions = [n for n in ast.walk(source) if isinstance(n, ast.FunctionDef) and n.name in names]
                self.assertEqual({n.name for n in functions}, names)
                for function in functions:
                    function.decorator_list = []
                attempts = []

                def failing_telemetry(event, **kwargs):
                    attempts.append(event)
                    raise RuntimeError("cannot start new thread")

                scope = dict(_ml_telemetry_async=failing_telemetry,
                             time=types.SimpleNamespace(time=lambda: 123),
                             PROG={"stage": "queued"}, JOBS={"job": {}}, log=lambda message: None,
                             request=types.SimpleNamespace(path="/api/generate", method="POST"))
                exec(compile(ast.Module(body=functions, type_ignores=[]), filename, "exec"), scope)
                scope["_set_job_stage"]("job", "loading")
                self.assertEqual(scope["PROG"]["stage"], "loading")
                self.assertEqual(scope["JOBS"]["job"]["stage_started"], 123)
                response = types.SimpleNamespace(status_code=402)
                self.assertIs(scope["_ml_report_api_failure"](response), response)
                scope["_ml_observe"]("notebook_h3_setup_completed")
                self.assertEqual(len(attempts), 3)
                response.status_code = 200
                self.assertIs(scope["_ml_report_api_failure"](response), response)
                self.assertEqual(len(attempts), 3)
                response.status_code = 500
                scope["request"].path = "/api/ml/activity"
                scope["_ml_report_api_failure"](response)
                self.assertEqual(len(attempts), 3)  # No telemetry-error reporting loop.


if __name__ == "__main__":
    unittest.main()

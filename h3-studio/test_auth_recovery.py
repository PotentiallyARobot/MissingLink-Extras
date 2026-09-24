"""Exercise authentication control flow without loading CUDA or model weights."""
import ast
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

TREE = ast.parse(Path(__file__).with_name('h3_studio_3.py').read_text(encoding='utf-8'))

def function(name, ns):
    node = next(n for n in ast.walk(TREE) if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<auth>', 'exec'), ns)
    return ns[name]

class AuthRecoveryTests(unittest.TestCase):
    def test_failure_backoff_force_retry_and_positive_ttl(self):
        state = {'ok': False, 'checked': 90, 'error': 'denied'}
        clock = Mock(return_value=100)
        upstream = Mock(return_value=(False, 'denied'))
        ns = dict(_ML_AUTH_STATE=state, _ML_AUTH_CHECK_LOCK=threading.Lock(),
                  _ml_time=SimpleNamespace(monotonic=clock), _validate_missinglink_token_uncached=upstream)
        validate = function('_validate_missinglink_token', ns)
        self.assertFalse(validate()[0]); self.assertEqual(upstream.call_count, 1)
        for _ in range(100): self.assertFalse(validate()[0])
        self.assertEqual(upstream.call_count, 1)
        validate(force=True); self.assertEqual(upstream.call_count, 2)
        state.update(ok=True, checked=90, error=''); upstream.return_value=(True, '')
        validate(); self.assertEqual(state['checked'], 90)  # Polls must never extend positive TTL.

    def test_access_recovery_validates_in_endpoint_not_gate(self):
        validate = Mock(return_value=(False, 'denied'))
        ns = dict(request=SimpleNamespace(path='/api/ml/access'), _validate_missinglink_token=validate)
        self.assertIsNone(function('_missinglink_ui_gate', ns)())
        validate.assert_not_called()
        ns.update(jsonify=lambda **kw:kw, MISSING_LINK_SIGNIN_URL='https://example.test')
        response = function('api_ml_access', ns)()
        self.assertEqual(response[1], 401)
        validate.assert_called_once_with(force=True)

    def test_rejected_generation_recorded_server_side_without_prompt(self):
        observe = Mock()
        ns = dict(request=SimpleNamespace(path='/api/generate', method='POST'),
                  _ml_observe=observe, _ml_uuid=SimpleNamespace(uuid4=lambda:SimpleNamespace(hex='request-1')))
        response=SimpleNamespace(status_code=401, get_json=lambda **kw:{'code':'missinglink_auth_required'})
        self.assertIs(function('_ml_report_api_failure', ns)(response), response)
        event=observe.call_args_list[0]
        self.assertEqual(event.args[0], 'notebook_h3_generation_rejected')
        self.assertEqual(event.kwargs['meta']['request_id'], 'request-1')
        self.assertNotIn('prompt', event.kwargs['meta'])
        observe.reset_mock(); ns['request'].path='/api/queue'
        ns['_ml_report_api_failure'](response)
        self.assertEqual(observe.call_count, 1)
        self.assertEqual(observe.call_args.args[0], 'notebook_h3_api_failed')

if __name__ == '__main__': unittest.main()

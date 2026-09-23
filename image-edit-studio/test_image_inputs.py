"""CPU regression checks; does not import torch or download model weights."""
import ast
import base64
import io
import json
import queue
import threading
import time
import types
import unittest
import uuid
from pathlib import Path
from unittest.mock import Mock
from PIL import Image, ImageFilter
from image_inputs import decode_image, validate_inputs, output_size, resize_image


def encoded(image, **kwargs):
    buffer = io.BytesIO()
    image.save(buffer, **kwargs)
    return base64.b64encode(buffer.getvalue()).decode()


class InputTests(unittest.TestCase):
    def test_phone_orientation(self):
        image = Image.new('RGB', (24, 12))
        exif = image.getexif()
        exif[274] = 6
        self.assertEqual(decode_image(encoded(image, format='JPEG', exif=exif)).size, (12, 24))

    def test_invalid_image_and_mask_are_rejected(self):
        valid = encoded(Image.new('RGB', (10, 20)), format='PNG')
        for body in [None, {'images': {}}, {'images': {'0': '%%%'}},
                     {'images': {'0': valid}, 'prompt': 'edit', 'mask': '%%%'}]:
            with self.subTest(body=body), self.assertRaises(ValueError):
                validate_inputs(body)

    def test_invalid_settings_are_rejected(self):
        valid = encoded(Image.new('RGB', (10, 20)), format='PNG')
        for key, value in [('width', -1), ('height', 9000), ('num_inference_steps', 0),
                           ('num_images_per_prompt', 100), ('seed', 'no'),
                           ('true_cfg_scale', float('nan')), ('resize_mode', 'unknown')]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_inputs({'images': {'0': valid}, 'prompt': 'edit', key: value})

    def test_aspect_preserving_and_explicit_modes(self):
        image = Image.new('RGB', (800, 400), 'red')
        size = output_size(image, {})
        self.assertAlmostEqual(size[0] / size[1], 2, delta=.1)
        padded = resize_image(image, (512, 512))
        self.assertEqual(padded.getpixel((256, 0)), (0, 0, 0))
        self.assertEqual(padded.getpixel((256, 256)), (255, 0, 0))
        for mode in ('crop', 'stretch'):
            self.assertEqual(resize_image(image, (512, 512), mode).getpixel((256, 0)), (255, 0, 0))


class WorkerTests(unittest.TestCase):
    def setUp(self):
        # Execute the real worker functions with a fake pipeline, leaving model startup untouched.
        tree = ast.parse(Path(__file__).with_name('server.py').read_text(encoding='utf-8'))
        names = {'GenerationCancelled', 'check_cancel', 'JobEvents', 'get_queue_position',
                 'get_queue_length', 'img_to_b64', 'b64_to_img', 'run_generation_blocking',
                 'api_generate', 'api_cancel'}
        selected = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.name in names]
        for node in selected:
            node.decorator_list = []
        self.ns = dict(queue=queue, threading=threading, time=time, uuid=uuid,
                       io=io, base64=base64, json=json, Image=Image, ImageFilter=ImageFilter,
                       decode_image=decode_image, validate_inputs=validate_inputs,
                       resize_image=resize_image, output_size=output_size,
                       _job_cancel={}, _gen_progress={}, wait_list=[], history=[],
                       wait_list_lock=threading.Lock(), gpu_sem=threading.Semaphore(1),
                       status={'ready': True}, MODEL_CONFIG={}, torch=Mock(), Request=object)
        exec(compile(ast.Module(body=selected, type_ignores=[]), 'server.py', 'exec'), self.ns)
        self.ns['pipeline'] = Mock(return_value=types.SimpleNamespace(images=[Image.new('RGB', (64, 64))]))

    def run_job(self, cancel=False, jid='test'):
        event = threading.Event()
        if cancel: event.set()
        self.ns['_job_cancel'][jid] = event
        events = self.ns['JobEvents'](jid)
        self.ns['run_generation_blocking'](jid, {'prompt': 'edit', 'seed': -1}, [Image.new('RGB', (64, 64))], events)
        self.assertNotIn(jid, self.ns['_job_cancel'])
        self.assertEqual(self.ns['wait_list'], [])
        self.assertEqual(self.ns['gpu_sem']._value, 1)
        return self.ns['_gen_progress'][jid]

    def test_cancel_before_gpu_acquisition(self):
        self.assertEqual(self.run_job(cancel=True)['type'], 'cancelled')
        self.ns['pipeline'].assert_not_called()

    def test_running_cancel_cleans_up_and_next_job_succeeds(self):
        hook = Mock()
        self.ns['pipeline']._all_hooks = [hook]
        def sampling(**kwargs):
            self.ns['_job_cancel']['test'].set()
            kwargs['callback_on_step_end'](None, 0, None, {})
        self.ns['pipeline'].side_effect = sampling
        self.assertEqual(self.run_job()['type'], 'cancelled')
        hook.offload.assert_called_once()
        self.assertEqual(self.ns['history'], [])
        self.ns['pipeline'].side_effect = None
        done = self.run_job(jid='next')
        self.assertEqual(done['type'], 'done')
        self.assertEqual(len(done['results']), 1)
        self.assertEqual(len(self.ns['history']), 1)

    def test_bad_upload_returns_client_error_without_starting_worker(self):
        import asyncio
        class Request:
            async def json(self): return {'images': {'0': '%%%'}}
        self.ns['StreamingResponse'] = lambda stream, **kwargs: types.SimpleNamespace(stream=stream, **kwargs)
        response = asyncio.run(self.ns['api_generate'](Request()))
        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.ns['_job_cancel'], {})
        self.ns['pipeline'].assert_not_called()


if __name__ == '__main__':
    unittest.main()

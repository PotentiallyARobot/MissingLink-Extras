"""Run with python -m unittest discover -s h3-studio -p test_h3_swaps.py."""
import io
import ast
import base64
import json
import os
import tempfile
import threading
import time
import unittest
import urllib.request
import urllib.error
import uuid
from unittest.mock import patch
from pathlib import Path

from flask import Flask
from PIL import Image, ImageDraw

from h3_swaps import api_mask, composite, inject_swaps, png, prepare_mask, register_swaps


class SwapTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.original = Image.new("RGB", (48, 32), (10, 20, 30))
        self.mask = Image.new("L", self.original.size)
        ImageDraw.Draw(self.mask).rectangle((12, 8, 35, 23), fill=255)
        self.key = "test-key"
        self.requests = []
        self.gate = None
        self.provider_fail = False
        def renderer(**kwargs):
            self.requests.append(kwargs)
            if self.gate:
                self.gate.wait(2)
            if self.provider_fail:
                raise RuntimeError("Provider rejected this edit")
            with Image.open(kwargs["mask_path"]) as uploaded_mask:
                alpha = uploaded_mask.getchannel("A")
            self.assertEqual(alpha.getextrema(), (0, 255))
            self.assertEqual(len(kwargs["reference_paths"]), 2)
            with Image.open(kwargs["reference_paths"][0]) as first:
                self.assertEqual(first.size, alpha.size)
            return png(Image.new("RGB", alpha.size, "red")), "", True
        self.app = Flask(__name__)
        register_swaps(self.app, output_dir=lambda:self.temp.name, api_key=lambda:self.key,
                       image_request=renderer, segmenter=lambda image,prompt:[(self.mask, .9)])
        self.client = self.app.test_client()

    def data(self, **changes):
        result = dict(original=(io.BytesIO(png(self.original)), "source.png"),
                      reference=(io.BytesIO(png(self.original)), "ref.png"),
                      mask=(io.BytesIO(png(self.mask)), "mask.png"), kind="face", grow="0", feather="0")
        result.update(changes)
        return result

    def wait(self, response):
        self.assertEqual(response.status_code, 202, response.json)
        for _ in range(200):
            result = self.client.get("/api/swaps/jobs/" + response.json["job"]).json
            if result["status"] != "running":
                return result
            time.sleep(.01)
        self.fail("Job did not finish")

    def test_mask_polarity_and_feather_preserve_outside(self):
        mask = prepare_mask(self.mask, self.original.size, feather=3)
        self.assertEqual(api_mask(mask).getpixel((0,0))[3], 255)
        result = composite(self.original, Image.new("RGB", self.original.size, "red"), mask)
        for y in range(result.height):
            for x in range(result.width):
                if not self.mask.getpixel((x,y)):
                    self.assertEqual(result.getpixel((x,y)), self.original.getpixel((x,y)))

    def test_edit_roundtrip_and_exact_unmasked_pixels(self):
        result = self.wait(self.client.post("/api/swaps/edit", data=self.data()))
        self.assertEqual(result["status"], "done", result)
        with Image.open(Path(self.temp.name)/result["file"]) as edited:
            self.assertEqual(edited.size,self.original.size)
            self.assertEqual(edited.getpixel((0,0)), (10,20,30))
            self.assertEqual(edited.getpixel((24,16)), (255,0,0))

    def test_empty_mask_rejected_without_api_call(self):
        r=self.client.post("/api/swaps/edit",data=self.data(mask=(io.BytesIO(png(Image.new('L',(48,32)))), 'mask.png')))
        self.assertEqual(r.status_code,400)
        self.assertFalse(self.requests)

    def test_mismatched_mask_and_invalid_options(self):
        r=self.client.post("/api/swaps/edit",data=self.data(mask=(io.BytesIO(png(Image.new('L',(4,4),255))), 'mask.png')))
        self.assertEqual(r.status_code,400)
        for changes in ({"grow":"-1"},{"feather":"33"},{"kind":"video"},{"quality":"bad"}):
            self.assertEqual(self.client.post('/api/swaps/edit',data=self.data(**changes)).status_code,400)
        self.assertFalse(self.requests)

    def test_missing_key_and_cross_origin(self):
        self.key=""
        self.assertEqual(self.client.post('/api/swaps/edit',data=self.data()).status_code,400)
        self.key="test-key"
        self.assertEqual(self.client.post('/api/swaps/edit',data=self.data(),headers={"Origin":"https://elsewhere.test"}).status_code,400)
        self.assertFalse(self.requests)

    def test_provider_error_and_busy_release(self):
        self.provider_fail=True
        result=self.wait(self.client.post('/api/swaps/edit',data=self.data()))
        self.assertEqual(result['status'],'error')
        self.provider_fail=False
        self.assertEqual(self.wait(self.client.post('/api/swaps/edit',data=self.data()))['status'],'done')

    def test_overlapping_edit_rejected(self):
        self.gate=threading.Event()
        first=self.client.post('/api/swaps/edit',data=self.data())
        try:
            self.assertEqual(self.client.post('/api/swaps/edit',data=self.data()).status_code,409)
        finally:
            self.gate.set()
        self.assertEqual(self.wait(first)['status'],'done')

    def test_sam_candidates_and_assets(self):
        r=self.wait(self.client.post('/api/swaps/mask',data=self.data(target='face')))
        self.assertEqual(len(r['candidates']),1)
        with self.client.get('/swaps/panel.html') as response:
            self.assertEqual(response.status_code,200)
        self.assertEqual(self.client.get('/swaps/secret.py').status_code,404)
        self.assertEqual(self.client.get('/api/swaps/jobs/missing').status_code,404)

    def test_tab_integration(self):
        page='<button id=tab_ref2va class=modetab type=button>REF2VA · REFERENCES</button></body>'
        result=inject_swaps(page)
        self.assertIn('id="tab_swaps"',result)
        self.assertIn('assignOutFileToFrame',result)
        self.assertIn('e.origin!==location.origin',result)

    def test_real_image_adapter_sends_mask_and_two_images(self):
        tree=ast.parse(Path(__file__).with_name('h3_studio_3.py').read_text(encoding='utf-8'))
        names={'_multipart_body','_openai_image_request'}
        functions=[node for node in ast.walk(tree) if isinstance(node,ast.FunctionDef) and node.name in names]
        namespace=dict(os=os,io=io,Image=Image,uuid=uuid,json=json,base64=base64,urllib=urllib)
        exec(compile(ast.Module(body=functions,type_ignores=[]),'image_adapter','exec'),namespace)
        paths=[str(Path(self.temp.name)/name) for name in ('first.png','second.png','mask.png')]
        self.original.save(paths[0]);self.original.save(paths[1]);api_mask(self.mask).save(paths[2])
        response=io.BytesIO(json.dumps({'data':[{'b64_json':base64.b64encode(png(self.original)).decode()}]}).encode())
        with patch('urllib.request.urlopen',return_value=response) as request:
            namespace['_openai_image_request'](key='test',model='gpt-image-2.5-sunburst',prompt='edit',width=48,height=32,
                quality='medium',reference_paths=paths[:2],mask_path=paths[2])
        sent=request.call_args.args[0]
        self.assertEqual(sent.full_url,'https://api.openai.com/v1/images/edits')
        self.assertEqual(sent.data.count(b'name="image[]"'),2)
        self.assertIn(b'name="mask"; filename="mask.png"',sent.data)


if __name__ == '__main__':
    unittest.main()

import sys
import requests
import unittest
from typing import Any

from image_utils import *
from const_map import *


class TestApi(unittest.TestCase):
    def test_detection_api(self):
        url = "http://localhost:8301/detect"
        self.body, self.image0 = self.create_request_body()

        content = self.check_response(url)

        self.check_content(content)

    def test_detect_batch_api(self):
        url = "http://localhost:8301/detect-batch"
        self.body, self.image0 = self.create_request_body()

        content = self.check_response(url)

        self.check_content(content)

    def test_uniformity_2_endpoints(self):
        url_detect = "http://localhost:8301/detect"
        url_detect_batch = "http://localhost:8301/detect-batch"

        self.body, self.image0 = self.create_request_body()

        content_detect = self.check_response(url_detect)
        content_detect_batch = self.check_response(url_detect_batch)

        self.assertEqual(content_detect, content_detect_batch)

    def create_request_body(self) -> tuple[dict[str, Any], Image.Image]:
        test_image_path = "test/image.jpg"
        image = read_image(test_image_path)
        resized_image = resize(image, new_shape=640)

        image0 = Image.fromarray(resized_image, "RGB")
        image0_area = image0.size[0] * image0.size[1]
        buffered = io.BytesIO()
        image0.save(buffered, format="PNG")

        byte64_str = image_to_base64(resized_image)
        revert_bytearr = base64_to_bytearr(byte64_str)
        self.assertEqual(buffered.getvalue(), revert_bytearr)

        body = {
            "session": "session",
            "base64_images": [byte64_str, byte64_str],
        }
        return body, image0

    def check_response(self, url: str) -> dict:
        res = requests.post(url, json=self.body)
        status_code = res.status_code
        self.assertEqual(status_code, 200)

        res = res.json()
        self.assertIsInstance(res, dict)
        self.assertEqual(len(res), 2)
        content = res["content"]

        return content

    def check_content(self, content):
        self.assertIsInstance(content, list)
        self.assertEqual(len(content), len(self.body["base64_images"]))
        for _ in content:
            self.assertIsInstance(_["raw"], dict)
            self.assertEqual(len(_["raw"]), 3)

            self.assertIsInstance(_["raw"]["detected_items"], list)
            for box in _["raw"]["detected_items"]:
                self.assertIsInstance(box, dict)
                self.assertEqual(len(box), 5)

                self.assertIsInstance(box["box"], dict)
                self.assertEqual(len(box["box"]), 4)
                self.assertTrue(0 <= box["box"]["left"] <= self.image0.size[0])
                self.assertTrue(0 <= box["box"]["top"] <= self.image0.size[1])
                self.assertTrue(0 <= box["box"]["right"] <= self.image0.size[0])
                self.assertTrue(0 <= box["box"]["bottom"] <= self.image0.size[1])

                self.assertIsInstance(box["category"], int)
                self.assertTrue(box["category"] in DETECT_CLOTHING_LABELS_EN.keys() and box["category"] not in reject_cate)

                self.assertIsInstance(box["label_en"], str)
                self.assertTrue(box["label_en"] == DETECT_CLOTHING_LABELS_EN[box["category"]])

                self.assertIsInstance(box["label_vn"], str)
                self.assertTrue(box["label_vn"] == DETECT_CLOTHING_LABELS_VN[box["category"]])

                self.assertIsInstance(box["area"], float)
                self.assertTrue(0 <= box["area"] <= 1)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        sys.argv.pop()
    unittest.main(verbosity=2)

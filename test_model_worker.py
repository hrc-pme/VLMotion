import requests, json, base64, io
from PIL import Image
worker_addr = "http://127.0.0.1:22000"

img = Image.new("RGB", (224, 224), color = "red")
buffered = io.BytesIO()
img.save(buffered, format="JPEG", quality=85)
raw_base64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

pload = {
    "model": "llava-v1.5-13b",
    "prompt": "USER: <image>\nwho are you?\nASSISTANT:",
    "temperature": 0.5,
    "top_p": 0.9,
    "max_new_tokens": 512,
    "images": [raw_base64_str],
    "stop": "</s>",
    "do_sample": True,
    "task_type": "general"
}

response = requests.post(worker_addr + "/worker_generate_stream", json=pload, stream=True)
for chunk in response.iter_lines(chunk_size=8192, decode_unicode=False, delimiter=b"\0"):
    if chunk:
        print("Got chunk!")
        print(chunk.decode("utf-8"))

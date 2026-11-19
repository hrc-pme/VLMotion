#!/usr/bin/env python3
"""Run one dataset image through an already running VLPoint worker.

This makes an inference request only; it does not start a camera or command a robot.
Run inside the project container after the controller and worker are ready.
"""

import argparse
import json
from pathlib import Path

import requests
from PIL import Image

from point.conversation import conv_templates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", default="http://127.0.0.1:11000")
    parser.add_argument("--model", default="vla13")
    parser.add_argument(
        "--data-root", default="/workspace/training_vlmotion_100",
        type=Path,
    )
    args = parser.parse_args()

    records = json.loads((args.data_root / "split" / "validation.json").read_text())
    record = records[0]
    image_path = args.data_root / "images" / record["image"]
    image = Image.open(image_path).convert("RGB")
    question = record["conversations"][0]["value"]
    expected = record["conversations"][1]["value"]

    conv = conv_templates["llava_v1"].copy()
    conv.append_message(conv.roles[0], (question, image, "Pad"))
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    _, images, _ = conv.get_images()

    models_response = requests.post(args.controller + "/list_models", timeout=10)
    models_response.raise_for_status()
    models = models_response.json().get("models", [])
    if args.model not in models:
        raise SystemExit(f"Model {args.model!r} is not registered; available: {models}")

    address_response = requests.post(
        args.controller + "/get_worker_address",
        json={"model": args.model}, timeout=10,
    )
    address_response.raise_for_status()
    worker_address = address_response.json().get("address")
    if not worker_address:
        raise SystemExit(f"Controller has no worker address for {args.model!r}")
    try:
        status_response = requests.post(
            worker_address + "/worker_get_status", json={}, timeout=5,
        )
        status_response.raise_for_status()
    except requests.RequestException as exc:
        raise SystemExit(
            f"Worker {worker_address} is not reachable. Start worker.launch.py "
            f"and keep it running before this test. Details: {exc}"
        ) from exc

    request = {
        "model": args.model,
        "prompt": prompt,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_new_tokens": 128,
        "stop": conv.sep2,
        "images": images,
    }
    print(f"Image: {image_path}")
    print(f"Question: {question.splitlines()[1]}")
    print(f"Annotated answer: {expected}")
    print(f"Worker: {worker_address}")

    answer = None
    try:
        with requests.post(
            worker_address + "/worker_generate_stream",
            json=request, stream=True, timeout=(10, 300),
        ) as response:
            response.raise_for_status()
            for chunk in response.iter_lines(delimiter=b"\0"):
                if not chunk:
                    continue
                data = json.loads(chunk)
                if data.get("error_code", 1) != 0:
                    raise SystemExit(f"Worker error: {data}")
                answer = data.get("text", "").removeprefix(prompt).strip()
    except requests.RequestException as exc:
        raise SystemExit(f"Inference request failed: {exc}") from exc

    if answer is None:
        raise SystemExit("Worker returned no response")
    print(f"Model answer: {answer}")


if __name__ == "__main__":
    main()

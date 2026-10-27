import base64
import json
import re
import time
from pathlib import Path

import requests
from PIL import Image


WORKER_ADDR = "http://127.0.0.1:22000"
MODEL = "llava-v1.5-13b"
IMAGE_DIR = Path("/workspace/便利商店")
OUTPUT_DIR = Path("/workspace/Outputs/convenience_store_vlm")

QUESTIONS = [
    (
        "Level-1",
        "這個人會不會靠近？",
        "Will the main person approach the camera/viewer?",
        "請從「會靠近／不會靠近／無法判斷」選一個，並加一句很短理由。",
    ),
    (
        "Level-2",
        "這個人將往哪個方向移動？",
        "Which direction will the main person move next?",
        "請從「往鏡頭靠近／遠離鏡頭／往左／往右／停留原地／無法判斷」選一個，並加一句很短理由。",
    ),
    (
        "Level-3",
        "這個人在做什麼？",
        "What is the main person doing?",
        "請用一句話描述主要動作。",
    ),
    (
        "Level-4",
        "這個人距離我很近嗎？",
        "Is the main person very close to the camera/viewer?",
        "請從「很近／不近／無法判斷」選一個，並加一句很短理由。",
    ),
    (
        "Level-5",
        "這個人的注意力在哪裡？",
        "Where is the main person's attention focused?",
        "請從「商品或貨架／收銀台或機台／鏡頭或我／手機／其他人／無法判斷」選一個，並加一句很短理由。",
    ),
]


def image_sort_key(path: Path):
    match = re.search(r"\((\d+)\)", path.name)
    return int(match.group(1)) if match else path.name


def encode_image(path: Path) -> str:
    with Image.open(path) as img:
        img = img.convert("RGB")
        tmp_path = OUTPUT_DIR / f"{path.stem}.jpg"
        img.save(tmp_path, format="JPEG", quality=85)
        data = tmp_path.read_bytes()
        tmp_path.unlink(missing_ok=True)
    return base64.b64encode(data).decode("utf-8")


def clean_answer(text: str) -> str:
    for marker in ("ASSISTANT:", "答案：", "Answer:"):
        if marker in text:
            text = text.split(marker)[-1]
    text = text.strip()
    text = re.sub(r"^(答案[:：]\s*)+", "", text).strip()
    return " ".join(text.split())


def ask_vlm(image_b64: str, level: str, question_en: str, instruction: str) -> str:
    prompt = (
        "USER: <image>\n"
        "Analyze this first-person image taken inside a convenience store. "
        "Focus only on the main visible person. Use only visual evidence in the image. "
        "If the answer cannot be inferred from the image, say 無法判斷. "
        "Answer in Traditional Chinese only. Do not repeat the question or these instructions. "
        "Start with 答案： and keep the answer to one concise sentence.\n"
        f"{level} question: {question_en}\n"
        f"Output rule: {instruction}\n"
        "ASSISTANT: 答案："
    )
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "temperature": 0.2,
        "top_p": 0.9,
        "max_new_tokens": 80,
        "images": [image_b64],
        "stop": "</s>",
        "do_sample": False,
        "task_type": "general",
    }
    response = requests.post(
        WORKER_ADDR + "/worker_generate_stream",
        json=payload,
        stream=True,
        timeout=120,
    )
    response.raise_for_status()

    last_text = ""
    for chunk in response.iter_lines(chunk_size=8192, decode_unicode=False, delimiter=b"\0"):
        if not chunk:
            continue
        data = json.loads(chunk.decode("utf-8"))
        if data.get("error_code", 0) != 0:
            raise RuntimeError(data)
        last_text = data.get("text", last_text)

    return clean_answer(last_text)


def write_outputs(results):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    jsonl_path = OUTPUT_DIR / "results.jsonl"
    json_path = OUTPUT_DIR / "results.json"
    md_path = OUTPUT_DIR / "results.md"

    with jsonl_path.open("w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    json_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = ["# Convenience Store VLM Results", ""]
    for item in results:
        lines.append(f"## {item['image']}")
        for answer in item["answers"]:
            lines.append(f"- {answer['level']} {answer['question']['zh']} {answer['answer']}")
        lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return jsonl_path, json_path, md_path


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(IMAGE_DIR.glob("*.png"), key=image_sort_key)
    results = []

    for index, path in enumerate(image_paths, start=1):
        print(f"[{index}/{len(image_paths)}] {path.name}", flush=True)
        image_b64 = encode_image(path)
        item = {"image": path.name, "path": str(path), "answers": []}

        for level, question_zh, question_en, instruction in QUESTIONS:
            started = time.time()
            answer = ask_vlm(image_b64, level, question_en, instruction)
            elapsed = time.time() - started
            print(f"  {level}: {answer} ({elapsed:.1f}s)", flush=True)
            item["answers"].append(
                {
                    "level": level,
                    "question": {"zh": question_zh, "en": question_en},
                    "answer": answer,
                    "elapsed_seconds": round(elapsed, 2),
                }
            )
        results.append(item)
        write_outputs(results)

    jsonl_path, json_path, md_path = write_outputs(results)
    print(f"Wrote {jsonl_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()

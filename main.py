

import base64
import requests
from config import HF_API_KEY

API_URL = "https://router.huggingface.co/v1/chat/completions"

HEADERS = {
    "Authorization": f"Bearer {HF_API_KEY}",
    "Content-Type": "application/json"
}

MODELS = [
    "zai-org/GLM-4.5V",
    "Qwen/Qwen2.5-VL-72B-Instruct",
    "Qwen/Qwen2.5-VL-32B-Instruct",
    "google/gemma-3-27b-it",
]


def data_url(image_bytes: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode("utf-8")


def extract_error(response: requests.Response) -> str:
    try:
        data = response.json()
        return data.get("error", {}).get("message", "") or str(data)
    except Exception:
        return (response.text or "").strip() or response.reason or "Request failed"


def box(title: str, lines: list[str], icon: str):
    width = max(30, len(title) + 4, *(len(line) for line in lines))

    print("\n" + "┏" + "━" * (width + 2) + "┓")
    print(f"┃ {icon} {title.ljust(width - 2)} ┃")
    print("┣" + "━" * (width + 2) + "┫")

    for line in lines:
        print(f"┃ {line.ljust(width)} ┃")

    print("┗" + "━" * (width + 2) + "┛\n")


def generate_caption():
    image_file = input(
        "Enter image filename (default: image.jpg): "
    ).strip() or "image.jpg"

    try:
        with open(image_file, "rb") as f:
            image_bytes = f.read()
    except Exception as e:
        box("Error", [str(e)], "❌")
        return

    payload_base = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Give a short caption for this image."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_url(image_bytes)
                        }
                    }
                ]
            }
        ],
        "max_tokens": 60,
        "temperature": 0.2
    }

    last_error = None

    for model in MODELS:
        print(f"Trying model: {model}")

        payload = dict(payload_base, model=model)

        try:
            response = requests.post(
                API_URL,
                headers=HEADERS,
                json=payload,
                timeout=120
            )
        except requests.RequestException as e:
            last_error = f"Request failed: {e}"
            continue

        if response.status_code != 200:
            last_error = extract_error(response)
            continue

        try:
            data = response.json()
        except Exception:
            last_error = f"Invalid JSON response:\n{response.text}"
            continue

        caption = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )

        if caption:
            box(
                "Image Caption Generated",
                [
                    f"Image   : {image_file}",
                    f"Model   : {model}",
                    f"Caption : {caption}"
                ],
                "✅"
            )
            return

    box(
        "Caption Failed",
        [
            f"Image : {image_file}",
            f"Error : {last_error or 'Unknown error'}"
        ],
        "❌"
    )


def main():
    generate_caption()


if __name__ == "__main__":
    main()

import os, base64, requests
from config import HF_API_KEY

API_URL = "https://router.huggingface.co/v1/chat/completions"
HEADERS = {"Authorization": f"Bearer {HF_API_KEY}", "Content-Type": "application/json"}

MODELS = [
    "Qwen/Qwen3-VL-8B-Instruct:together",
    "Qwen/Qwen3-VL-32B-Instruct:together",
    "Qwen/Qwen2.5-VL-32B-Instruct:together",
    "Qwen/Qwen2-VL-7B-Instruct:together",
]

def data_url(b: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(b).decode("utf-8")

def extract_err(r: requests.Response) -> str:
    try:
        j = r.json()
        return j.get("error", {}).get("message") or str(j)
    except Exception:
        return (r.text or "").strip() or r.reason or "Request failed."

def query_caption(image_bytes: bytes) -> tuple[str | None, str | None]:
    base = {
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Give a short caption for this image."},
                {"type": "image_url", "image_url": {"url": data_url(image_bytes)}},
            ],
        }],
        "max_tokens": 60,
        "temperature": 0.2,
    }
    last_err = None
    for model in MODELS:
        try:
            r = requests.post(API_URL, headers=HEADERS, json={**base, "model": model}, timeout=120)
        except requests.RequestException as e:
            last_err = f"Request failed: {e}"
            continue
        if r.status_code != 200:
            last_err = extract_err(r)
            continue
        try:
            d = r.json()
        except Exception:
            last_err = "Non-JSON response received from the API."
            continue
        cap = (d.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()
        if cap:
            return cap, None
        last_err = "No caption found."
    return None, last_err or "Unknown error"

def main():
    folder_path = input("Enter the path to your images folder (press Enter for 'images'): ").strip() or "images"
    if not os.path.isdir(folder_path):
        print(f"Folder '{folder_path}' does not exist. Exiting.")
        return

    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not image_files:
        print(f"No valid image files found in '{folder_path}'. Exiting.")
        return

    captions = []
    for img_name in image_files:
        img_path = os.path.join(folder_path, img_name)
        print(f"\nProcessing: {img_path}")

        try:
            with open(img_path, "rb") as img_file:
                image_bytes = img_file.read()
        except Exception as e:
            print(f"Could not load image '{img_name}'. Error: {e}")
            continue

        cap, err = query_caption(image_bytes)
        if err:
            print(f"[API Error] {err} for '{img_name}'")
            continue

        print(f"Caption: {cap}")
        captions.append((img_name, cap))

    if captions:
        summary_file = os.path.join(folder_path, "captions_summary.txt")
        with open(summary_file, "w", encoding="utf-8") as sf:
            for img_name, caption in captions:
                sf.write(f"{img_name}: {caption}\n")
        print(f"\nAll captions saved to: {summary_file}")
    else:
        print("\nNo captions were generated. Please check for errors or try different images.")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import os
import urllib.request
import xml.etree.ElementTree as ET
import tempfile

RSS_URL = "https://anchor.fm/s/212aa0c8/podcast/rss"
OUTPUT_DIR = "transcripts"
MODEL_NAME = os.getenv("WHISPER_MODEL", "base")

def format_timestamp(seconds: float) -> str:
    total_ms = int(round(seconds * 1000))
    hours = total_ms // 3600000
    minutes = (total_ms % 3600000) // 60000
    secs = (total_ms % 60000) // 1000
    ms = total_ms % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"

def fetch_rss_items():
    print(f"📡 Obteniendo feed RSS desde: {RSS_URL}")
    req = urllib.request.Request(RSS_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        xml_data = resp.read()

    root = ET.fromstring(xml_data)
    channel = root.find("channel")
    if channel is None:
        return []

    items = []
    for item in channel.findall("item"):
        title = item.findtext("title", default="Sin título").strip()
        guid_elem = item.find("guid")
        guid = guid_elem.text.strip() if guid_elem is not None and guid_elem.text else None
        enclosure = item.find("enclosure")
        audio_url = enclosure.get("url") if enclosure is not None else None

        if guid and audio_url:
            items.append({"title": title, "guid": guid, "audio_url": audio_url})
    return items

def transcribe_audio(audio_path: str, output_vtt_path: str):
    from faster_whisper import WhisperModel
    print(f"🤖 Cargando modelo Whisper ({MODEL_NAME})...")
    model = WhisperModel(MODEL_NAME, device="cpu", compute_type="int8")

    print(f"🎙️ Transcribiendo audio con IA...")
    segments, _ = model.transcribe(audio_path, language="es", beam_size=5)

    with open(output_vtt_path, "w", encoding="utf-8") as f:
        f.write("WEBVTT - Manzanas Enfrentadas Transcripción Oficial\n\n")
        idx = 1
        for seg in segments:
            text = seg.text.strip()
            if text:
                f.write(f"{idx}\n")
                f.write(f"{format_timestamp(seg.start)} --> {format_timestamp(seg.end)}\n")
                f.write(f"{text}\n\n")
                idx += 1
    print(f"✅ Guardado en: {output_vtt_path}")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    items = fetch_rss_items()
    print(f"📋 Episodios en el feed: {len(items)}")

    to_process = []
    for item in items[:3]:
        vtt_path = os.path.join(OUTPUT_DIR, f"{item['guid'].lower()}.vtt")
        if not os.path.exists(vtt_path):
            to_process.append((item, vtt_path))

    if not to_process:
        print("🎉 Todo al día.")
        return

    for item, vtt_path in to_process:
        print(f"\n--- Transcribiendo: {item['title']} ---")
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio:
            temp_path = temp_audio.name

        try:
            req = urllib.request.Request(item['audio_url'], headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as resp, open(temp_path, "wb") as out_f:
                out_f.write(resp.read())

            transcribe_audio(temp_path, vtt_path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

if __name__ == "__main__":
    main()

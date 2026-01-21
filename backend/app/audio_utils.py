import httpx
import tempfile
import os
from app.config import GROQ_API_KEY


class AudioError(Exception):
    pass

GROQ_WHISPER_MODEL = "whisper-large-v3"


async def extract_text_from_audio(file_bytes: bytes) -> str:
    if not GROQ_API_KEY:
        raise AudioError("GROQ_API_KEY is not configured")

    tmp_path = None

    try:
        # Save audio to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
        }

        data = {
            "model": GROQ_WHISPER_MODEL,
        }

        # IMPORTANT: open file inside context manager
        with open(tmp_path, "rb") as audio_file:
            files = {
                "file": ("audio.webm", audio_file, "audio/webm"),
            }

            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    "https://api.groq.com/openai/v1/audio/transcriptions",
                    headers=headers,
                    data=data,
                    files=files,
                )

        if response.status_code != 200:
            raise AudioError(
                f"Groq Whisper error {response.status_code}: {response.text}"
            )

        result = response.json()
        text = result.get("text", "").strip()

        if not text:
            raise AudioError("No speech detected in audio")

        return text

    except AudioError:
        raise
    except Exception as e:
        raise AudioError(str(e))

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

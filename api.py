# OpenAI API Spec. Reference: https://platform.openai.com/docs/api-reference/audio/createSpeech

from contextlib import asynccontextmanager
from io import BytesIO
import gc
import hashlib
import os
import time

import torch
import torchaudio
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from g2pw import G2PWConverter
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

from cosyvoice.utils.file_utils import load_wav
from single_inference import CustomCosyVoice, get_bopomofo_rare


class Settings(BaseSettings):
    api_key: str = Field(
        default="", description="Specifies the API key used to authenticate the user."
    )

    model_path: str = Field(
        default="MediaTek-Research/BreezyVoice",
        description="Specifies the model used for speech synthesis.",
    )
    speaker_prompt_audio_path: str = Field(
        default="./data/example.wav",
        description="Specifies the path to the prompt speech audio file of the speaker.",
    )
    llm_fp16: bool = Field(
        default=False,
        description="Run the LLM stage under CUDA fp16 autocast (~15% faster, audio not bit-identical). Env: LLM_FP16=1.",
    )
    speaker_prompt_text_transcription: str = Field(
        default="在密碼學中，加密是將明文資訊改變為難以讀取的密文內容，使之不可讀的方法。只有擁有解密方法的對象，經由解密過程，才能將密文還原為正常可讀的內容。",
        description="Specifies the transcription of the speaker prompt audio.",
    )


class SpeechRequest(BaseModel):
    model: str = ""
    input: str = Field(
        description="The content that will be synthesized into speech. You can include phonetic symbols if needed, though they should be used sparingly.",
        examples=["今天天氣真好"],
    )
    input_file: str = ""
    response_format: str = ""
    speed: float = 1.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.settings = Settings()
    prompt_text_path = os.environ.get("SPEAKER_PROMPT_TEXT_TRANSCRIPTION_FILE")
    if prompt_text_path:
        with open(prompt_text_path, "r", encoding="utf-8") as file:
            app.state.settings.speaker_prompt_text_transcription = file.read().strip()
    app.state.cosyvoice = CustomCosyVoice(app.state.settings.model_path)
    app.state.bopomofo_converter = G2PWConverter()
    app.state.prompt_speech_16k = load_wav(
        app.state.settings.speaker_prompt_audio_path, 16000
    )
    # The speaker prompt never changes while the server runs, so analyse it once
    # instead of once per sentence (saves ~1.4 s of CPU work per request).
    app.state.prompt_cache = app.state.cosyvoice.build_prompt_cache(
        app.state.settings.speaker_prompt_text_transcription,
        app.state.prompt_speech_16k,
    )
    print(f"[startup] prompt cached, llm_fp16={app.state.settings.llm_fp16}", flush=True)
    yield
    del app.state.cosyvoice
    del app.state.bopomofo_converter


app = FastAPI(lifespan=lifespan, root_path="/v1")


@app.get("/models")
async def get_models(request: Request):
    return {
        "object": "list",
        "data": [
            {
                "id": request.app.state.settings.model_path,
                "object": "model",
                "created": 0,
                "owned_by": "local",
            }
        ],
    }


@app.post("/audio/speech")
async def speach_endpoint(request: Request, payload: SpeechRequest):
    input_text = payload.input
    if payload.input_file:
        with open(payload.input_file, "r", encoding="utf-8") as file:
            input_text = file.read().strip()
    request_id = hashlib.sha1(input_text.encode("utf-8")).hexdigest()[:10]
    preview = input_text[:80].replace("\n", " ")
    start = time.time()
    print(
        f"[speech:{request_id}] start chars={len(input_text)} input_file={payload.input_file!r} text={preview!r}",
        flush=True,
    )
    try:
        output = request.app.state.cosyvoice.inference_zero_shot_cached(
            input_text,
            request.app.state.prompt_cache,
            llm_fp16=request.app.state.settings.llm_fp16,
        )
        elapsed = time.time() - start
        print(f"[speech:{request_id}] inference_done sec={elapsed:.2f}", flush=True)
        audio_buffer = BytesIO()
        torchaudio.save(audio_buffer, output["tts_speech"], 22050, format="wav")
        print(f"[speech:{request_id}] wav_ready bytes={audio_buffer.getbuffer().nbytes}", flush=True)
        audio_buffer.seek(0)
        return StreamingResponse(
            audio_buffer,
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=output.wav"},
        )
    except Exception as exc:
        elapsed = time.time() - start
        print(
            f"[speech:{request_id}] error sec={elapsed:.2f} type={type(exc).__name__}: {exc}",
            flush=True,
        )
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc
    finally:
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception as cleanup_exc:
                print(
                    f"[speech:{request_id}] cuda_cleanup_error type={type(cleanup_exc).__name__}: {cleanup_exc}",
                    flush=True,
                )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8080)

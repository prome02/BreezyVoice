"""Tests for reusing the analysed voice sample across synthesis requests.

Run inside WSL with the BreezyVoice venv:
    .venv/bin/python -m unittest tests.test_prompt_cache -v
"""
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from single_inference import CustomCosyVoice  # noqa: E402


class FakeFrontend:
    """Deterministic stand-in for CustomCosyVoiceFrontEnd that counts prompt analysis."""

    def __init__(self):
        self.prompt_analysis_calls = 0

    def text_normalize(self, text, split=False):
        text = text.strip()
        if not split:
            return text
        return [piece + "。" for piece in text.split("。") if piece]

    def _extract_text_token(self, text):
        token = torch.tensor([[ord(ch) % 1000 for ch in text]], dtype=torch.int32)
        return token, torch.tensor([token.shape[1]], dtype=torch.int32)

    def frontend_zero_shot(self, tts_text, prompt_text, prompt_speech_16k):
        self.prompt_analysis_calls += 1
        text_token, text_token_len = self._extract_text_token(tts_text)
        prompt_token, prompt_token_len = self._extract_text_token(prompt_text)
        speech_token = (prompt_speech_16k[:, :5] * 100).to(torch.int32)
        speech_feat = prompt_speech_16k[:, :8].reshape(1, -1, 1).repeat(1, 1, 80)
        embedding = torch.full((1, 192), float(prompt_speech_16k.sum()))
        return {
            "text": text_token, "text_len": text_token_len,
            "prompt_text": prompt_token, "prompt_text_len": prompt_token_len,
            "llm_prompt_speech_token": speech_token,
            "llm_prompt_speech_token_len": torch.tensor([speech_token.shape[1]], dtype=torch.int32),
            "flow_prompt_speech_token": speech_token,
            "flow_prompt_speech_token_len": torch.tensor([speech_token.shape[1]], dtype=torch.int32),
            "prompt_speech_feat": speech_feat,
            "prompt_speech_feat_len": torch.tensor([speech_feat.shape[1]], dtype=torch.int32),
            "llm_embedding": embedding, "flow_embedding": embedding,
        }


class FakeModel:
    def __init__(self):
        self.calls = []

    def inference(self, **model_input):
        self.calls.append(model_input)
        n = int(model_input["text_len"][0])
        return {"tts_speech": torch.ones(1, n * 10)}


def make_tts():
    tts = CustomCosyVoice.__new__(CustomCosyVoice)  # skip loading real weights
    tts.frontend = FakeFrontend()
    tts.model = FakeModel()
    return tts


PROMPT_TEXT = "這是一段聲音樣本的逐字稿。"
PROMPT_WAV = torch.linspace(0.1, 0.9, 16000).reshape(1, -1)


def assert_inputs_equal(test, actual, expected):
    test.assertEqual(set(actual), set(expected))
    for key in expected:
        test.assertTrue(torch.equal(actual[key], expected[key]), key)


class PromptCacheTest(unittest.TestCase):
    def test_cache_analyses_voice_sample_once_and_drops_text(self):
        tts = make_tts()
        cache = tts.build_prompt_cache(PROMPT_TEXT, PROMPT_WAV)
        self.assertEqual(tts.frontend.prompt_analysis_calls, 1)
        self.assertNotIn("text", cache)
        self.assertNotIn("text_len", cache)

    def test_cached_requests_never_reanalyse_voice_sample(self):
        tts = make_tts()
        cache = tts.build_prompt_cache(PROMPT_TEXT, PROMPT_WAV)
        for text in ["第一句。第二句。", "第三句。", "第四句。第五句。第六句。"]:
            tts.inference_zero_shot_cached(text, cache)
        self.assertEqual(tts.frontend.prompt_analysis_calls, 1)
        self.assertEqual(len(tts.model.calls), 6)

    def test_cached_inputs_match_original_zero_shot_inputs(self):
        cached_tts = make_tts()
        cache = cached_tts.build_prompt_cache(PROMPT_TEXT, PROMPT_WAV)
        cached_tts.inference_zero_shot_cached("他回頭看了一眼。她沒有說話。", cache)

        original_tts = make_tts()
        original_tts.inference_zero_shot("他回頭看了一眼。她沒有說話。", PROMPT_TEXT, PROMPT_WAV)

        self.assertEqual(len(cached_tts.model.calls), len(original_tts.model.calls))
        for actual, expected in zip(cached_tts.model.calls, original_tts.model.calls):
            assert_inputs_equal(self, actual, expected)

    def test_output_concatenates_every_piece(self):
        tts = make_tts()
        cache = tts.build_prompt_cache(PROMPT_TEXT, PROMPT_WAV)
        result = tts.inference_zero_shot_cached("一二三。四五。", cache)
        self.assertEqual(result["tts_speech"].shape, (1, (4 + 3) * 10))


class RecordingStage:
    """Fake llm/flow/hift stage that records whether CUDA fp16 autocast was on."""

    def __init__(self, name, log, output):
        self.name, self.log, self.output = name, log, output

    def inference(self, **kwargs):
        self.log.append((self.name, torch.is_autocast_enabled()))
        return self.output(kwargs)


class StagedModel:
    def __init__(self):
        self.device = torch.device("cpu")
        self.log = []
        self.llm = RecordingStage("llm", self.log, lambda kw: torch.ones(1, 7, dtype=torch.int64))
        self.flow = RecordingStage("flow", self.log, lambda kw: torch.ones(1, 80, 14))
        self.hift = RecordingStage("hift", self.log, lambda kw: torch.ones(1, 140))

    def inference(self, **model_input):
        raise AssertionError("fp16 path must run the stages itself")


@unittest.skipUnless(torch.cuda.is_available(), "autocast check needs CUDA")
class LlmFp16Test(unittest.TestCase):
    def test_fp16_applies_to_llm_stage_only(self):
        tts = CustomCosyVoice.__new__(CustomCosyVoice)
        tts.frontend = FakeFrontend()
        tts.model = StagedModel()
        cache = tts.build_prompt_cache(PROMPT_TEXT, PROMPT_WAV)
        result = tts.inference_zero_shot_cached("第一句。", cache, llm_fp16=True)
        self.assertEqual(tts.model.log, [("llm", True), ("flow", False), ("hift", False)])
        self.assertEqual(result["tts_speech"].shape, (1, 140))

    def test_default_keeps_original_full_precision_path(self):
        tts = make_tts()
        cache = tts.build_prompt_cache(PROMPT_TEXT, PROMPT_WAV)
        tts.inference_zero_shot_cached("第一句。", cache)
        self.assertEqual(len(tts.model.calls), 1)


if __name__ == "__main__":
    unittest.main()

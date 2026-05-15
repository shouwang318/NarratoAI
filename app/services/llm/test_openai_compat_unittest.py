"""OpenAI 兼容 provider 的最小回归测试。"""

import asyncio
import unittest
from unittest.mock import patch

from app.config import config
from app.services.llm.base import TextModelProvider
from app.services.llm.manager import LLMServiceManager
from app.services.llm.migration_adapter import LegacyLLMAdapter, VisionAnalyzerAdapter
from app.services.llm.openai_compatible_provider import OpenAICompatibleVisionProvider
from app.services.llm.providers import register_all_providers


class DummyOpenAITextProvider(TextModelProvider):
    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def supported_models(self) -> list[str]:
        return []

    async def generate_text(self, prompt: str, **kwargs) -> str:
        return prompt

    async def _make_api_call(self, payload: dict) -> dict:
        return payload


class DummyNoKeyTextProvider(DummyOpenAITextProvider):
    requires_api_key = False

    @property
    def provider_name(self) -> str:
        return "codex"


def _reset_manager_state():
    LLMServiceManager._vision_providers.clear()
    LLMServiceManager._text_providers.clear()
    LLMServiceManager._vision_instance_cache.clear()
    LLMServiceManager._text_instance_cache.clear()


class OpenAICompatManagerTests(unittest.TestCase):
    def setUp(self):
        _reset_manager_state()
        self._original_app = dict(config.app)

    def tearDown(self):
        _reset_manager_state()
        config.app.clear()
        config.app.update(self._original_app)

    def test_register_all_providers_registers_openai_and_codex_providers(self):
        register_all_providers()

        self.assertEqual({"openai", "codex"}, set(LLMServiceManager.list_text_providers()))
        self.assertEqual({"openai", "codex"}, set(LLMServiceManager.list_vision_providers()))

    def test_get_text_provider_uses_openai_keys(self):
        LLMServiceManager.register_text_provider("openai", DummyOpenAITextProvider)

        config.app["text_llm_provider"] = "openai"
        config.app["text_openai_api_key"] = "new-key"
        config.app["text_openai_model_name"] = "new-model"
        config.app["text_openai_base_url"] = "https://new.example/v1"

        provider = LLMServiceManager.get_text_provider()

        self.assertIsInstance(provider, DummyOpenAITextProvider)
        self.assertEqual("new-key", provider.api_key)
        self.assertEqual("new-model", provider.model_name)
        self.assertEqual("https://new.example/v1", provider.base_url)

    def test_get_text_provider_allows_no_key_provider(self):
        LLMServiceManager.register_text_provider("codex", DummyNoKeyTextProvider)

        config.app["text_llm_provider"] = "codex"
        config.app["text_codex_api_key"] = ""
        config.app["text_codex_model_name"] = "gpt-5.4"

        provider = LLMServiceManager.get_text_provider()

        self.assertIsInstance(provider, DummyNoKeyTextProvider)
        self.assertEqual("", provider.api_key)
        self.assertEqual("gpt-5.4", provider.model_name)


class OpenAICompatVisionConcurrencyTests(unittest.IsolatedAsyncioTestCase):
    async def test_analyze_images_keeps_batch_order_when_running_concurrently(self):
        provider = OpenAICompatibleVisionProvider(api_key="k", model_name="m")
        provider._prepare_images = lambda images: list(images)

        async def fake_analyze_batch(batch, prompt, **kwargs):
            delays = {"a": 0.03, "c": 0.01, "e": 0.0}
            await asyncio.sleep(delays[batch[0]])
            return f"batch-{batch[0]}"

        provider._analyze_batch = fake_analyze_batch

        result = await provider.analyze_images(
            images=["a", "b", "c", "d", "e", "f"],
            prompt="prompt",
            batch_size=2,
            max_concurrency=2,
        )

        self.assertEqual(["batch-a", "batch-c", "batch-e"], result)

    async def test_analyze_images_respects_max_concurrency_limit(self):
        provider = OpenAICompatibleVisionProvider(api_key="k", model_name="m")
        provider._prepare_images = lambda images: list(images)

        in_flight = 0
        max_in_flight = 0

        async def fake_analyze_batch(batch, prompt, **kwargs):
            nonlocal in_flight, max_in_flight
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
            await asyncio.sleep(0.02)
            in_flight -= 1
            return f"batch-{batch[0]}"

        provider._analyze_batch = fake_analyze_batch

        result = await provider.analyze_images(
            images=["a", "b", "c", "d", "e", "f"],
            prompt="prompt",
            batch_size=1,
            max_concurrency=2,
        )

        self.assertEqual(6, len(result))
        self.assertEqual(2, max_in_flight)


class ExplicitVisionAdapterSettingsTests(unittest.IsolatedAsyncioTestCase):
    class _CapturingVisionProvider:
        last_init: tuple[str, str, str | None] | None = None
        last_call_kwargs: dict | None = None

        def __init__(self, api_key: str, model_name: str, base_url: str | None = None):
            self.api_key = api_key
            self.model_name = model_name
            self.base_url = base_url
            ExplicitVisionAdapterSettingsTests._CapturingVisionProvider.last_init = (api_key, model_name, base_url)

        async def analyze_images(self, images, prompt, batch_size=10, max_concurrency=1, **kwargs):
            ExplicitVisionAdapterSettingsTests._CapturingVisionProvider.last_call_kwargs = dict(kwargs)
            return [f"{self.model_name}|{self.api_key}|{self.base_url}"]

    def setUp(self):
        _reset_manager_state()
        self._original_app = dict(config.app)

    def tearDown(self):
        _reset_manager_state()
        config.app.clear()
        config.app.update(self._original_app)

    async def test_adapter_uses_explicit_settings_instead_of_global_config(self):
        LLMServiceManager.register_vision_provider("openai", self._CapturingVisionProvider)
        config.app["vision_openai_api_key"] = "config-key"
        config.app["vision_openai_model_name"] = "config-model"
        config.app["vision_openai_base_url"] = "https://config.example/v1"

        adapter = VisionAnalyzerAdapter(
            provider="openai",
            api_key="explicit-key",
            model="explicit-model",
            base_url="https://explicit.example/v1",
        )
        result = await adapter.analyze_images(
            images=["/tmp/keyframe_000001_000000100.jpg"],
            prompt="描述画面",
            batch_size=1,
            max_concurrency=1,
        )

        self.assertEqual(
            ("explicit-key", "explicit-model", "https://explicit.example/v1"),
            self._CapturingVisionProvider.last_init,
        )
        self.assertEqual("explicit-key", self._CapturingVisionProvider.last_call_kwargs["api_key"])
        self.assertEqual("https://explicit.example/v1", self._CapturingVisionProvider.last_call_kwargs["api_base"])
        self.assertEqual("explicit-model|explicit-key|https://explicit.example/v1", result[0]["response"])


class LegacyNarrationAdapterBehaviorTests(unittest.TestCase):
    def test_generate_narration_returns_raw_unrecoverable_payload_without_fabrication(self):
        raw_payload = "not-json-at-all ::: ???"

        with patch(
            "app.services.llm.migration_adapter.PromptManager.get_prompt",
            return_value="prompt",
        ), patch(
            "app.services.llm.migration_adapter._run_async_safely",
            return_value=raw_payload,
        ):
            result = LegacyLLMAdapter.generate_narration(
                markdown_content="markdown",
                api_key="test-key",
                base_url="https://example.com/v1",
                model="test-model",
            )

        self.assertEqual(raw_payload, result)
        self.assertNotIn('"items"', result)


if __name__ == "__main__":
    unittest.main()

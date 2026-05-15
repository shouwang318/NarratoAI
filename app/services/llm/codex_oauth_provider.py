"""
Codex OAuth backed LLM provider.

NarratoAI delegates generation to `codex exec --ephemeral`. Codex owns the
ChatGPT OAuth flow and token refresh; NarratoAI never reads Codex credential
files or stores OAuth tokens.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import PIL.Image
from loguru import logger

from app.config import config
from app.config.defaults import DEFAULT_CODEX_COMMAND
from .base import TextModelProvider, VisionModelProvider
from .exceptions import APICallError, AuthenticationError


def _resolve_codex_command(command: str | None = None) -> str:
    configured = (command or config.app.get("codex_command") or DEFAULT_CODEX_COMMAND).strip()
    expanded = os.path.expanduser(configured)
    if os.path.isabs(expanded) and os.path.exists(expanded):
        return expanded

    resolved = shutil.which(expanded)
    if resolved:
        return resolved

    fallback = os.path.expanduser("~/.local/bin/codex")
    if os.path.exists(fallback):
        return fallback

    return expanded


def _build_generation_prompt(
    prompt: str,
    system_prompt: Optional[str] = None,
    response_format: Optional[str] = None,
) -> str:
    parts = [
        "You are the LLM backend for NarratoAI.",
        "Return only the requested content.",
        "Do not inspect files, run shell commands, edit files, or include implementation commentary.",
    ]
    if response_format == "json":
        parts.append("Return strict JSON only, with no Markdown fences or surrounding prose.")
    if system_prompt:
        parts.extend(["", "System instructions:", system_prompt])
    parts.extend(["", "User request:", prompt])
    return "\n".join(parts)


class _CodexExecClient:
    def __init__(
        self,
        model_name: str,
        timeout_seconds: float,
        command: str | None = None,
        cwd: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds
        self.command = _resolve_codex_command(command)
        self.cwd = cwd or str(Path.cwd())

    async def run_text(self, text: str) -> str:
        return await self.run_turn(prompt=text, image_paths=[])

    async def run_turn(self, prompt: str, image_paths: list[str]) -> str:
        with tempfile.TemporaryDirectory(prefix="narratoai-codex-") as temp_dir:
            output_path = Path(temp_dir) / "last-message.txt"
            args = [
                self.command,
                "exec",
                "--ephemeral",
                "--sandbox",
                "read-only",
                "--model",
                self.model_name,
                "--skip-git-repo-check",
                "--output-last-message",
                str(output_path),
            ]
            for image_path in image_paths:
                args.extend(["--image", image_path])
            args.append("-")

            try:
                process = await asyncio.create_subprocess_exec(
                    *args,
                    cwd=self.cwd,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            except FileNotFoundError as exc:
                raise AuthenticationError(
                    f"Codex command not found: {self.command}. Install Codex or set app.codex_command."
                ) from exc

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(prompt.encode("utf-8")),
                    timeout=self.timeout_seconds,
                )
            except asyncio.TimeoutError as exc:
                process.kill()
                await process.wait()
                raise APICallError(f"Codex exec timed out after {self.timeout_seconds} seconds") from exc
            if process.returncode != 0:
                detail = "\n".join(
                    part.decode("utf-8", errors="replace").strip()
                    for part in (stdout, stderr)
                    if part
                ).strip()
                if "not logged in" in detail.lower() or "login" in detail.lower():
                    raise AuthenticationError(
                        "Codex is not logged in. Run `codex login` and choose ChatGPT OAuth."
                    )
                raise APICallError(detail or f"Codex exec failed with exit code {process.returncode}")

            if output_path.exists():
                result = output_path.read_text(encoding="utf-8").strip()
                if result:
                    return result

            fallback = stdout.decode("utf-8", errors="replace").strip()
            if fallback:
                return fallback
            raise APICallError("Codex completed without an agent response")


class _CodexOAuthBase:
    requires_api_key = False

    @property
    def provider_name(self) -> str:
        return "codex"

    @property
    def supported_models(self) -> List[str]:
        return []

    def _validate_model_support(self):
        logger.debug(f"Codex OAuth model configured: {self.model_name}")

    def _build_client(self, timeout_seconds: float) -> _CodexExecClient:
        return _CodexExecClient(
            model_name=self.model_name,
            timeout_seconds=timeout_seconds,
            command=config.app.get("codex_command"),
        )


class CodexOAuthTextProvider(_CodexOAuthBase, TextModelProvider):
    """Text generation through Codex ChatGPT OAuth."""

    async def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        response_format: Optional[str] = None,
        **kwargs,
    ) -> str:
        del temperature, max_tokens, kwargs
        timeout_seconds = float(config.app.get("llm_text_timeout", 180))
        client = self._build_client(timeout_seconds)
        codex_prompt = _build_generation_prompt(prompt, system_prompt, response_format)
        return await client.run_text(codex_prompt)

    async def _make_api_call(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return payload


class CodexOAuthVisionProvider(_CodexOAuthBase, VisionModelProvider):
    """Vision analysis through Codex ChatGPT OAuth."""

    async def analyze_images(
        self,
        images: List[Union[str, Path, PIL.Image.Image]],
        prompt: str,
        batch_size: int = 10,
        max_concurrency: int = 1,
        **kwargs,
    ) -> List[str]:
        del kwargs
        bounded_concurrency = max(1, int(max_concurrency))
        batches = [images[index : index + batch_size] for index in range(0, len(images), batch_size)]
        semaphore = asyncio.Semaphore(bounded_concurrency)

        async def run_batch(batch: list[Union[str, Path, PIL.Image.Image]]) -> str:
            async with semaphore:
                return await self._analyze_batch(batch, prompt)

        return await asyncio.gather(*(run_batch(batch) for batch in batches))

    async def _analyze_batch(self, batch: list[Union[str, Path, PIL.Image.Image]], prompt: str) -> str:
        timeout_seconds = float(config.app.get("llm_vision_timeout", 120))
        client = self._build_client(timeout_seconds)
        with tempfile.TemporaryDirectory(prefix="narratoai-codex-images-") as temp_dir:
            image_paths = self._build_local_image_paths(batch, Path(temp_dir))
            return await client.run_turn(prompt=_build_generation_prompt(prompt), image_paths=image_paths)

    def _build_local_image_paths(
        self,
        images: list[Union[str, Path, PIL.Image.Image]],
        temp_dir: Path,
    ) -> list[str]:
        paths: list[str] = []
        for index, image in enumerate(images):
            if isinstance(image, PIL.Image.Image):
                image_path = temp_dir / f"image-{index}.jpg"
                image.convert("RGB").save(image_path, format="JPEG", quality=85)
            else:
                image_path = Path(image).expanduser().resolve()
            paths.append(str(image_path))
        return paths

    async def _make_api_call(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return payload

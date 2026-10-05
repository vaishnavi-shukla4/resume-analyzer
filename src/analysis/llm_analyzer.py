"""Groq LLM analyzer with structured JSON validation and no import-time client."""

from __future__ import annotations

import json
from typing import Any, Optional

from pydantic import ValidationError

from prompts.resume_prompts import resume_job_match_messages
from src.models.schemas import (
    AnalysisError,
    JobDescription,
    LLMMatchResult,
    LLMStructuredOutput,
    ResumeDocument,
)
from src.utils.config_loader import get_groq_api_key, load_dotenv_file


def llm_json_schema() -> dict[str, Any]:
    schema = LLMStructuredOutput.model_json_schema()
    schema.setdefault("additionalProperties", False)
    schema.pop("title", None)
    return schema


class LLMAnalyzer:
    def __init__(
        self,
        config: dict[str, Any],
        api_key: Optional[str] = None,
        client: Any = None,
    ) -> None:
        load_dotenv_file()
        llm_cfg = config.get("llm")
        if not isinstance(llm_cfg, dict):
            raise KeyError("missing_config: llm")
        self.provider = str(llm_cfg.get("provider", "groq"))
        self.primary_model = llm_cfg.get("model")
        self.fallback_model = llm_cfg.get("fallback_model")
        if not self.primary_model or not self.fallback_model:
            raise KeyError("missing_config: llm.model / llm.fallback_model")
        self.temperature = float(llm_cfg.get("temperature", 0.2))
        self.max_tokens = int(llm_cfg.get("max_tokens", 2048))
        self._api_key = api_key if api_key is not None else get_groq_api_key()
        self._client = client
        self._startup_error: Optional[AnalysisError] = None
        if client is None and not self._api_key:
            self._startup_error = AnalysisError(
                category="missing_api_key",
                message="GROQ_API_KEY is not set. Traditional NLP still runs. LLM analysis is unavailable.",
                details="Create a project-root .env file with GROQ_API_KEY.",
            )

    def is_available(self) -> bool:
        return self._startup_error is None and (self._client is not None or bool(self._api_key))

    def availability_error(self) -> Optional[AnalysisError]:
        return self._startup_error

    def analyze(self, resume: ResumeDocument, job: JobDescription) -> LLMMatchResult:
        if self._startup_error is not None:
            return LLMMatchResult.unavailable(self._startup_error)
        try:
            client = self._ensure_client()
        except Exception as exc:
            return LLMMatchResult.unavailable(self._map_exception(exc))
        messages = resume_job_match_messages(resume.cleaned_text, job.cleaned_text)
        models = [str(self.primary_model), str(self.fallback_model)]
        last_error: Optional[AnalysisError] = None
        for model_id in models:
            result, last_error = self._analyze_with_model(client, model_id, messages)
            if result is not None:
                return result
            if last_error and last_error.category in {"invalid_api_key", "missing_api_key"}:
                break
        return LLMMatchResult.unavailable(
            last_error
            or AnalysisError(
                category="malformed_response",
                message="LLM analysis failed after retries.",
            )
        )

    def _analyze_with_model(
        self,
        client: Any,
        model_id: str,
        messages: list[dict[str, str]],
    ) -> tuple[Optional[LLMMatchResult], Optional[AnalysisError]]:
        last_error: Optional[AnalysisError] = None
        for _attempt in range(2):
            try:
                raw_text = self._complete(client, model_id, messages)
                payload = self._parse_structured(raw_text)
                return LLMMatchResult.from_structured(payload, model_used=model_id), None
            except ValidationError as exc:
                last_error = AnalysisError(
                    category="malformed_response",
                    message="The LLM returned JSON that did not match the required schema.",
                    details=str(exc),
                )
            except (json.JSONDecodeError, ValueError) as exc:
                last_error = AnalysisError(
                    category="malformed_response",
                    message="The LLM returned text that was not valid structured JSON.",
                    details=str(exc),
                )
            except Exception as exc:
                mapped = self._map_exception(exc)
                if mapped.category == "model_unavailable":
                    return None, mapped
                if mapped.category in {"invalid_api_key", "rate_limit", "network_failure"}:
                    return None, mapped
                last_error = mapped
        return None, last_error

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        if not self._api_key:
            raise RuntimeError("missing api key")
        from groq import Groq

        self._client = Groq(api_key=self._api_key)
        return self._client

    def _response_format(self) -> dict[str, Any]:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "resume_job_match",
                "strict": True,
                "schema": llm_json_schema(),
            },
        }

    def _complete(self, client: Any, model_id: str, messages: list[dict[str, str]]) -> str:
        create = client.chat.completions.create
        base_kwargs: dict[str, Any] = {
            "model": model_id,
            "messages": messages,
            "temperature": self.temperature,
            "response_format": self._response_format(),
        }
        try:
            response = create(**base_kwargs, max_completion_tokens=self.max_tokens)
        except TypeError:
            response = create(**base_kwargs, max_tokens=self.max_tokens)
        except Exception as first_error:
            if self._looks_like_schema_unsupported(first_error):
                fallback_format = {"type": "json_object"}
                base_kwargs["response_format"] = fallback_format
                try:
                    response = create(**base_kwargs, max_completion_tokens=self.max_tokens)
                except TypeError:
                    response = create(**base_kwargs, max_tokens=self.max_tokens)
            else:
                raise
        choice = response.choices[0].message
        content = getattr(choice, "content", None)
        if not content:
            raise ValueError("empty LLM content")
        return str(content)

    @staticmethod
    def _looks_like_schema_unsupported(exc: Exception) -> bool:
        text = str(exc).lower()
        return "response_format" in text or "json_schema" in text or "schema" in text

    @staticmethod
    def _parse_structured(raw_text: str) -> LLMStructuredOutput:
        text = raw_text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:].strip()
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("JSON root must be an object")
        return LLMStructuredOutput.model_validate(data)

    @staticmethod
    def _map_exception(exc: Exception) -> AnalysisError:
        name = exc.__class__.__name__.lower()
        message = str(exc)
        lowered = message.lower()
        status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
        if "missing api key" in lowered or "api key" in lowered and "missing" in lowered:
            return AnalysisError(category="missing_api_key", message="GROQ_API_KEY is missing.", details=name)
        if status == 401 or "authentication" in name or "invalid api" in lowered or "unauthorized" in lowered:
            return AnalysisError(
                category="invalid_api_key",
                message="Groq rejected the API key. LLM analysis is unavailable.",
                details=name,
            )
        if status == 429 or "ratelimit" in name or "rate limit" in lowered:
            return AnalysisError(
                category="rate_limit",
                message="Groq rate limit reached. LLM analysis is unavailable.",
                details=name,
            )
        if status in {404, 400} and ("model" in lowered or "not found" in lowered):
            return AnalysisError(
                category="model_unavailable",
                message="The configured Groq model is unavailable.",
                details=name,
            )
        if "not found" in lowered and "model" in lowered:
            return AnalysisError(
                category="model_unavailable",
                message="The configured Groq model is unavailable.",
                details=name,
            )
        if "connection" in name or "timeout" in name or "network" in lowered or "connect" in lowered:
            return AnalysisError(
                category="network_failure",
                message="Could not reach the Groq API. LLM analysis is unavailable.",
                details=name,
            )
        return AnalysisError(
            category="malformed_response",
            message="The Groq request failed. LLM analysis is unavailable.",
            details=name,
        )

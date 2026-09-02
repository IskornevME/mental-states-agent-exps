import logging
import os
import time
from typing import Any, Dict, List, Mapping, Optional

import requests

from agent_eval.agents.base import BaseAgent


logger = logging.getLogger("agent_eval")


class SGLangChatAgent(BaseAgent):
    """Actor client for SGLang's OpenAI-compatible chat endpoint.

    The server is responsible for applying the model's native chat template.
    This is the same interaction style used by the current Qwen3 actor in QLASS: POST /v1/chat/completions with structured chat messages.
    """

    def __init__(
        self,
        config: Mapping[str, Any],
    ) -> None:
        super().__init__(config)

        self.server_address = str(config["server_address"]).rstrip("/")

        self.model_name = str(config["model_name"]).strip()

        if not self.server_address:
            raise ValueError("server_address must be non-empty")

        if not self.model_name:
            raise ValueError("model_name must be non-empty")

        # Optional explicit model id exposed by /v1/models.
        # Usually this can be left empty.
        self.api_model_name = str(config.get("api_model_name", "") or "").strip()

        self._resolved_api_model_name: Optional[str] = None

        # Generation parameters.
        self.max_new_tokens = int(config.get("max_new_tokens", 512))
        self.temperature = float(config.get("temperature", 0.0))
        self.top_p = float(config.get("top_p", 1.0))

        top_k = config.get("top_k")
        self.top_k = None if top_k is None else int(top_k)

        min_p = config.get("min_p")
        self.min_p = None if min_p is None else float(min_p)

        self.presence_penalty = float(config.get("presence_penalty", 0.0))
        self.frequency_penalty = float(config.get("frequency_penalty", 0.0))

        # Request/retry settings.
        self.request_timeout = float(config.get("request_timeout", 300.0))
        self.max_retries = int(config.get("max_retries", 3))
        self.retry_delay_seconds = float(config.get("retry_delay_seconds", 5.0))

        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")

        if self.request_timeout <= 0:
            raise ValueError("request_timeout must be positive")

        if self.max_retries <= 0:
            raise ValueError("max_retries must be positive")

        # Additional SGLang-specific request arguments, for example min_tokens.
        self.extra_request_body = dict(config.get("extra_request_body", {}) or {})

        self.api_key = str(
            config.get("api_key", os.environ.get("OPENAI_API_KEY", "")) or ""
        )

        self._session = requests.Session()

    def _headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json"
        }

        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        return headers

    def _resolve_model_name(self) -> str:
        """Resolve the actual model id exposed by SGLang.

        This is useful when SGLang is launched from a local checkpoint path:
        the server-side model id can then differ from the Hugging Face model id stored in the experiment config.
        """
        if self._resolved_api_model_name:
            return self._resolved_api_model_name

        if self.api_model_name:
            self._resolved_api_model_name = self.api_model_name
            return self._resolved_api_model_name

        resolved = self.model_name

        try:
            response = self._session.get(
                f"{self.server_address}/v1/models",
                headers=self._headers(),
                timeout=min(self.request_timeout, 30.0),
            )
            response.raise_for_status()

            response_payload = response.json()

            if not isinstance(response_payload, Mapping):
                raise ValueError("Unexpected /v1/models response")

            model_ids = [
                str(item.get("id", "")).strip()
                for item in response_payload.get(
                    "data",
                    [],
                )
                if isinstance(item, Mapping)
                and str(
                    item.get("id", "")
                ).strip()
            ]

            if self.model_name in model_ids:
                resolved = self.model_name

            else:
                basename = os.path.basename(
                    self.model_name.rstrip("/")
                )

                basename_matches = [
                    model_id
                    for model_id in model_ids
                    if os.path.basename(
                        model_id.rstrip("/")
                    )
                    == basename
                ]

                if len(basename_matches) == 1:
                    resolved = basename_matches[0]

                elif len(model_ids) == 1:
                    # Most experiments run exactly one model per server.
                    resolved = model_ids[0]

        except (
            requests.RequestException,
            ValueError,
            TypeError,
        ) as exc:
            logger.warning(
                "Could not resolve SGLang model id "
                "through /v1/models: %s. "
                "Using configured model_name=%r.",
                exc,
                self.model_name,
            )

        self._resolved_api_model_name = resolved
        return resolved

    @staticmethod
    def _normalize_messages(
        messages: List[dict],
    ) -> List[Dict[str, str]]:
        """Validate messages before sending them to the server."""
        if not messages:
            raise ValueError(
                "Agent received an empty message list"
            )

        normalized: List[Dict[str, str]] = []

        for index, message in enumerate(messages):
            if not isinstance(message, Mapping):
                raise TypeError(
                    f"Message {index} must be a mapping, "
                    f"got {type(message).__name__}"
                )

            role = str(message.get("role", "")).strip()

            if role not in {"system", "user", "assistant"}:
                raise ValueError(
                    f"Unsupported role at message {index}: {role!r}"
                )

            normalized.append(
                {
                    "role": role,
                    "content": str(message.get("content", "") or ""),
                }
            )

        return normalized

    def _build_payload(
        self,
        messages: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self._resolve_model_name(),
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_new_tokens,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "n": 1,
            "stream": False,
        }

        # SGLang extensions supported by the current QLASS client.
        if self.top_k is not None and self.top_k >= 0:
            payload["top_k"] = self.top_k

        if self.min_p is not None:
            payload["min_p"] = self.min_p

        # Do not allow arbitrary config fields to replace the core
        # communication contract.
        protected_keys = {
            "model",
            "messages",
            "stream",
        }

        collisions = (
            protected_keys
            & self.extra_request_body.keys()
        )

        if collisions:
            raise ValueError(
                f"extra_request_body may not override {sorted(collisions)}"
            )

        payload.update(self.extra_request_body)

        return payload

    def act(self, messages: List[dict]) -> str:
        """Generate one actor response."""
        prepared_messages = self._normalize_messages(messages)

        payload = self._build_payload(prepared_messages)

        last_error: Optional[BaseException] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._session.post(
                    f"{self.server_address}/v1/chat/completions",
                    headers=self._headers(),
                    json=payload,
                    timeout=self.request_timeout,
                )

                response.raise_for_status()

                response_payload = response.json()

                if not isinstance(response_payload, Mapping):
                    raise ValueError("Unexpected SGLang response")

                choices = response_payload.get("choices", [])

                if not choices:
                    raise ValueError("SGLang response contains no choices")

                choice = choices[0]

                if not isinstance(choice, Mapping):
                    raise ValueError("Invalid SGLang choice")

                message = choice.get("message", {})

                if not isinstance(message, Mapping):
                    raise ValueError("Invalid SGLang message object")

                content = message.get("content")

                if (content is None or not str(content).strip()):
                    raise ValueError("SGLang returned an empty assistant response")

                return str(content).strip()

            except (
                requests.RequestException,
                ValueError,
                KeyError,
                TypeError,
            ) as exc:
                last_error = exc

                logger.warning(
                    "SGLang request failed (attempt %d/%d): %s",
                    attempt,
                    self.max_retries,
                    exc,
                )

                if (attempt < self.max_retries and self.retry_delay_seconds > 0):
                    time.sleep(self.retry_delay_seconds)

        raise RuntimeError(
            f"SGLang request failed after {self.max_retries} attempts"
        ) from last_error

    def close(self) -> None:
        """Close the persistent HTTP session."""
        self._session.close()

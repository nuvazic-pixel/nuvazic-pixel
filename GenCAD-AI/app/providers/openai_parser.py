import json
from typing import Any

from openai import OpenAI

from app.config import LLMConfig
from app.domain.intent import ParsedEngineeringIntent
from app.parsers.base import EngineeringParser
from app.providers.prompt import SYSTEM_PROMPT


class OpenAIEngineeringParser(EngineeringParser):
    """
    Cloud reference adapter for v0.2.2.

    The model may produce only ParsedEngineeringIntent.
    It never produces EngineeringSpec and therefore has no authority
    to create DERIVED engineering values.
    """

    def __init__(
        self,
        config: LLMConfig | None = None,
        client: OpenAI | None = None,
    ):
        self.config = config or LLMConfig()
        self.client = client or OpenAI(
            timeout=self.config.timeout_seconds,
            max_retries=self.config.max_retries,
        )

    def parse(self, prompt: str) -> ParsedEngineeringIntent:
        schema = ParsedEngineeringIntent.model_json_schema()

        response = self.client.responses.create(
            model=self.config.model,
            instructions=SYSTEM_PROMPT,
            input=prompt,
            store=False,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "parsed_engineering_intent",
                    "description": (
                        "Explicitly extracted engineering facts and uncertainty. "
                        "No derived engineering values."
                    ),
                    "schema": schema,
                    "strict": True,
                }
            },
        )

        if not response.output_text:
            raise RuntimeError("LLM returned no structured output")

        payload: dict[str, Any] = json.loads(response.output_text)
        return ParsedEngineeringIntent.model_validate(payload)

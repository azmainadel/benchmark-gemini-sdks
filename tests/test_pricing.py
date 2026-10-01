"""Offline regression tests for Gemini token-to-cost accounting."""

from __future__ import annotations

import unittest

from app.pricing import estimate_cost


class PricingTests(unittest.TestCase):
    def estimate(self, sdk: str, output: int, reasoning: int | None):
        return estimate_cost(
            sdk=sdk,
            model="gemini-3.8-flash",
            input_tokens=142,
            output_tokens=output,
            reasoning_tokens=reasoning,
            thinking_mode="on",
        )

    def test_google_and_agno_add_separately_reported_thinking_tokens(self) -> None:
        for sdk, candidates in (("google_genai", 38), ("agno", 37)):
            with self.subTest(sdk=sdk):
                result = self.estimate(
                    sdk, candidates, 982 if sdk == "google_genai" else 983
                )
                self.assertEqual(result.billable_output_tokens, 1020)
                self.assertEqual(result.estimated_cost_usd, 0.0039315)

    def test_litellm_and_langchain_output_counts_already_include_thoughts(self) -> None:
        for sdk in ("litellm", "langchain"):
            with self.subTest(sdk=sdk):
                result = self.estimate(sdk, 1020, 985 if sdk == "litellm" else None)
                self.assertEqual(result.billable_output_tokens, 1020)
                self.assertEqual(result.estimated_cost_usd, 0.0039315)

    def test_missing_thinking_usage_is_not_underpriced(self) -> None:
        result = self.estimate("google_genai", 38, None)
        self.assertIsNone(result.estimated_cost_usd)
        self.assertEqual(result.cost_status, "missing_reasoning_usage")

    def test_unpriced_model_is_explicit(self) -> None:
        result = estimate_cost(
            sdk="google_genai",
            model="gemini-unknown",
            input_tokens=1,
            output_tokens=1,
            reasoning_tokens=0,
            thinking_mode="off",
        )
        self.assertIsNone(result.estimated_cost_usd)
        self.assertEqual(result.cost_status, "unpriced_model")


if __name__ == "__main__":
    unittest.main()

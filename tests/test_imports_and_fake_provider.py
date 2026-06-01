import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class ImportAndFakeProviderTests(unittest.TestCase):
    def test_src_layout_packages_are_discovered(self):
        from setuptools import find_packages

        packages = find_packages(where=str(SRC))
        self.assertIn("FrugalGPT", packages)
        self.assertIn("service", packages)

    def test_top_level_import_is_lazy(self):
        import FrugalGPT

        self.assertIn("GenerationParameter", FrugalGPT.__all__)
        from FrugalGPT import GenerationParameter

        params = GenerationParameter(max_tokens=12, temperature=0)
        self.assertEqual(params.max_tokens, 12)

    def test_config_loader_finds_service_info(self):
        from FrugalGPT.config import load_service_info

        service_info = load_service_info()
        self.assertIn("openaichat", service_info)
        self.assertIn("fake", service_info)
        self.assertIn("support-cheap", service_info["fake"])

    def test_fake_provider_returns_business_triage_json(self):
        from service.modelservice import GenerationParameter, make_model

        model = make_model("fake", "support-cheap")
        result = model.getcompletion(
            "Customer says the API is down and they cannot login.",
            genparams=GenerationParameter(max_tokens=80),
        )

        payload = json.loads(result["completion"])
        self.assertEqual(payload["category"], "technical")
        self.assertEqual(payload["urgency"], "high")
        self.assertTrue(payload["escalation"])
        self.assertGreater(result["cost"], 0)

    def test_llmvanilla_batch_uses_per_result_cost(self):
        from FrugalGPT.llmvanilla import LLMVanilla
        from service.modelservice import GenerationParameter

        engine = LLMVanilla(
            service_name=["fake/support-cheap"],
            cache_enabled=False,
            max_workers=3,
        )
        queries = [
            ("Refund request after duplicate charge", "billing", "1"),
            ("Need pricing for a larger contract", "sales", "2"),
            ("API integration returns an error", "technical", "3"),
        ]

        result = engine.get_completion_batch(
            queries,
            service_name="fake/support-cheap",
            use_db=False,
            genparams=GenerationParameter(max_tokens=80),
        )

        self.assertEqual(len(result), 3)
        self.assertTrue((result["cost"] > 0).all())
        categories = {json.loads(answer)["category"] for answer in result["answer"]}
        self.assertEqual(categories, {"billing", "sales", "technical"})


if __name__ == "__main__":
    unittest.main()

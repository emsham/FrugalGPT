import json
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def load_business_eval_module():
    module_path = ROOT / "examples" / "business_ticket_eval.py"
    spec = importlib.util.spec_from_file_location("business_ticket_eval", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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

    def test_business_ticket_eval_writes_reviewable_csv(self):
        module = load_business_eval_module()
        cheap_service = module.make_service("fake/support-cheap")
        strong_service = module.make_service("fake/support-strong")
        rows = [
            {
                "ticket_id": "T-1",
                "text": "Customer says login is down and they may cancel.",
                "expected_category": "technical",
                "expected_urgency": "high",
                "expected_escalation": "true",
            },
            {
                "ticket_id": "T-2",
                "text": "Customer asks how to update their notification email.",
                "expected_category": "general",
                "expected_urgency": "normal",
                "expected_escalation": "false",
            },
        ]

        output_rows, cascade_total, strong_only_total, baseline_cascade_total = module.evaluate_rows(
            rows,
            cheap_service,
            "fake/support-cheap",
            strong_service,
            "fake/support-strong",
            confidence_threshold=0.7,
            compare_strong_only=True,
        )

        self.assertEqual(len(output_rows), 2)
        self.assertGreater(cascade_total, 0)
        self.assertGreater(strong_only_total, 0)
        self.assertEqual(cascade_total, baseline_cascade_total)
        self.assertEqual(output_rows[0]["routed_to"], "strong")
        self.assertEqual(output_rows[1]["routed_to"], "cheap")
        self.assertEqual(output_rows[0]["status"], "ok")
        self.assertIn("review_decision", output_rows[0])

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "eval.csv"
            module.write_output(output_path, output_rows)
            self.assertIn("reviewer_notes", output_path.read_text())

    def test_business_ticket_eval_records_provider_failures(self):
        module = load_business_eval_module()
        strong_service = module.make_service("fake/support-strong")

        class BadService:
            def getcompletion(self, context, genparams):
                return {"completion": "not json", "cost": 0.01}

        rows = [{"ticket_id": "T-1", "text": "This row should fail."}]
        output_rows, cascade_total, strong_only_total, baseline_cascade_total = module.evaluate_rows(
            rows,
            BadService(),
            "fake/bad",
            strong_service,
            "fake/support-strong",
            confidence_threshold=0.7,
        )

        self.assertEqual(cascade_total, 0)
        self.assertEqual(strong_only_total, 0)
        self.assertEqual(baseline_cascade_total, 0)
        self.assertEqual(output_rows[0]["status"], "error")
        self.assertIn("did not return JSON", output_rows[0]["error_message"])

    def test_business_ticket_eval_resolves_presets(self):
        module = load_business_eval_module()

        class Args:
            provider_preset = "openai"
            cheap_provider = "fake/support-cheap"
            strong_provider = "fake/support-strong"

        cheap_provider, strong_provider = module.resolve_provider_names(Args())
        self.assertEqual(cheap_provider, "openaichat/gpt-4o-mini")
        self.assertEqual(strong_provider, "openaichat/gpt-4o")
        self.assertEqual(module.required_env_vars(cheap_provider, strong_provider), ["OPENAI_API_KEY"])


if __name__ == "__main__":
    unittest.main()

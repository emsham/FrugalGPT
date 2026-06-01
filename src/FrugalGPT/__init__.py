from .utils import formatdata, getservicename, help, loadcsvdata

__all__ = [
    "DataLoader",
    "GenerationParameter",
    "LLMCascade",
    "LLMforAll",
    "compute_score",
    "compute_score_full",
    "formatdata",
    "getservicename",
    "help",
    "loadcsvdata",
]


def __getattr__(name):
    if name == "DataLoader":
        from .dataloader import DataLoader

        return DataLoader
    if name == "GenerationParameter":
        from service.modelservice import GenerationParameter

        return GenerationParameter
    if name == "LLMCascade":
        try:
            from .llmcascade import LLMCascade
        except ImportError as exc:
            raise RuntimeError(
                "LLMCascade requires the scoring dependencies. "
                "Install them with `pip install FrugalGPT[scoring]`."
            ) from exc

        return LLMCascade
    if name == "LLMforAll":
        from .llmvanilla import LLMVanilla

        return LLMVanilla
    if name in {"compute_score", "compute_score_full"}:
        from .evaluate import compute_score, compute_score_full

        return {"compute_score": compute_score, "compute_score_full": compute_score_full}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

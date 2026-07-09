import json
import os
import pickle
import time
from importlib import import_module

from FrugalGPT.config import load_service_info


_TOKENIZER = None
_TOKENIZER_FAILED = False


def compute_cost(input_size, output_size, service_info):
    cost = service_info["cost_input"] * input_size + service_info["cost_fixed"]
    if output_size > service_info["fixed_size"]:
        cost += service_info["cost_output"] * (output_size - service_info["fixed_size"])
    return cost


def _import_optional(module_name, extra_name=None):
    try:
        return import_module(module_name)
    except ImportError as exc:
        package = extra_name or module_name
        raise RuntimeError(
            f"Optional dependency {module_name!r} is required for this provider. "
            f"Install it with `pip install FrugalGPT[{package}]` or install the package directly."
        ) from exc


def _require_env(name, provider):
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Please set {name} before using the {provider} provider.")
    return value


def count_tokens(text):
    """Count tokens without network/cache side effects by default.

    Set FRUGALGPT_USE_GPT2_TOKENIZER=1 to opt in to local GPT-2 tokenization.
    """
    global _TOKENIZER, _TOKENIZER_FAILED
    if text is None:
        return 0
    if os.environ.get("FRUGALGPT_USE_GPT2_TOKENIZER") != "1":
        return max(1, len(str(text).split()))
    if _TOKENIZER is None and not _TOKENIZER_FAILED:
        try:
            transformers = _import_optional("transformers", "scoring")
            _TOKENIZER = transformers.GPT2Tokenizer.from_pretrained(
                "gpt2", local_files_only=True
            )
        except Exception:
            _TOKENIZER_FAILED = True
    if _TOKENIZER is not None:
        return len(_TOKENIZER(str(text))["input_ids"])
    return max(1, len(str(text).split()))


class GenerationParameter(object):
    def __init__(
        self,
        max_tokens=100,
        temperature=0.1,
        stop=None,
        date="20230301",
        trial=0,
    ):
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.stop = ["\n"] if stop is None else stop
        self.date = date
        self.readable = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stop": self.stop,
            "date": date,
        }

    def get_dict(self):
        return dict(self.readable)


class ModelService:
    """Base class for a model provider."""

    def getcompletion(
        self,
        context,
        use_save=False,
        savepath="raw.pkl",
        genparams=GenerationParameter(),
    ):
        raise NotImplementedError


class APIModelProvider(ModelService):
    """Base class for REST-style model providers."""

    _CONFIG = load_service_info()
    _REQUEST_TIMEOUT = 60

    def getcompletion(
        self,
        context,
        use_save=False,
        savepath="raw.pkl",
        genparams=GenerationParameter(),
    ):
        endpoint = self._get_endpoint()
        req = self._request_format(context, genparams)
        self.context = context
        latency = 0

        if use_save:
            try:
                response = self.read_response(savepath)
            except Exception:
                time1 = time.time()
                response = self._api_call(endpoint, data=req, api_key=self._API_KEY)
                latency = time.time() - time1
                self.write_response(savepath)
        else:
            time1 = time.time()
            response = self._api_call(endpoint, data=req, api_key=self._API_KEY)
            latency = time.time() - time1

        result = self._response_format(response)
        result["cost"] = self._get_cost(context, result)
        result["latency"] = latency
        return result

    def read_response(self, path="test"):
        with open(path, "rb") as file:
            self.response = pickle.load(file)
        try:
            return self.response.json()
        except AttributeError:
            return self.response

    def write_response(self, path="test"):
        with open(path, "wb") as file:
            pickle.dump(self.response, file)
        try:
            return self.response.json()
        except AttributeError:
            return self.response

    def _request_format(self, context, genparams):
        raise NotImplementedError

    def _response_format(self, response):
        raise NotImplementedError

    def _api_call(self, endpoint, data, api_key, retries=3, retry_grace_time=2):
        requests = _import_optional("requests", "all")
        last_error = None
        for attempt in range(retries):
            try:
                response = requests.post(
                    endpoint,
                    headers={"Authorization": f"Bearer {api_key}"},
                    json=data,
                    timeout=self._REQUEST_TIMEOUT,
                )
                if response.status_code == 200:
                    self.response = response
                    return response.json()
                response.raise_for_status()
            except requests.exceptions.RequestException as exc:
                last_error = exc
            if attempt < retries - 1:
                time.sleep(retry_grace_time * (2**attempt))
        raise TimeoutError(f"API request failed after {retries} attempts") from last_error

    def _get_cost(self, context, completion):
        tk1, tk2 = self._get_io_tokens(context, completion)
        try:
            service_info = self._CONFIG[self._NAME][self._model]
        except KeyError as exc:
            raise KeyError(f"No service cost config for {self._NAME}/{self._model}") from exc
        return compute_cost(tk1, tk2, service_info)

    def _get_endpoint(self):
        return self._ENDPOINT.format(engine=self._model)


class OpenAIModelProvider(APIModelProvider):
    _ENDPOINT = os.environ.get(
        "OPENAI_ENDPOINT", "https://api.openai.com/v1/engines/{engine}/completions"
    )
    _NAME = "openai"

    def __init__(self, model):
        self._model = model
        self._API_KEY = _require_env("OPENAI_API_KEY", "OpenAI")

    def _request_format(self, context, genparams):
        if self._model in ["tƒd"]:
            tk1 = count_tokens(context)
            if tk1 + genparams.max_tokens >= 2047:
                context = context[-2048 + genparams.max_tokens :]
        return {
            "prompt": context,
            "echo": False,
            "max_tokens": genparams.max_tokens,
            "logprobs": 1,
            "temperature": genparams.temperature,
            "top_p": 1,
            "stop": genparams.stop,
        }

    def _response_format(self, response):
        return {"raw": response, "completion": response["choices"][0]["text"]}

    def _get_io_tokens(self, context, completion):
        raw = completion["raw"]
        return raw["usage"]["prompt_tokens"], raw["usage"].get("completion_tokens", 0)


class OpenAIChatModelProvider(APIModelProvider):
    _ENDPOINT = os.environ.get(
        "OPENAICHAT_ENDPOINT", "https://api.openai.com/v1/chat/completions"
    )
    _NAME = "openaichat"

    def __init__(self, model):
        self._model = model
        self._API_KEY = _require_env("OPENAI_API_KEY", "OpenAI")

    def _request_format(self, context, genparams):
        return {
            "messages": [{"content": context, "role": "user"}],
            "max_tokens": genparams.max_tokens,
            "temperature": genparams.temperature,
            "model": self._model,
        }

    def _response_format(self, response):
        return {
            "raw": response,
            "completion": response["choices"][0]["message"]["content"],
        }

    def _get_io_tokens(self, context, completion):
        raw = completion["raw"]
        return raw["usage"]["prompt_tokens"], raw["usage"].get("completion_tokens", 0)


class AI21ModelProvider(APIModelProvider):
    _ENDPOINT = os.environ.get(
        "AI21_STUDIO_ENDPOINT", "https://api.ai21.com/studio/v1/{engine}/complete"
    )
    _NAME = "ai21"

    def __init__(self, model):
        self._model = model
        self._API_KEY = _require_env("AI21_STUDIO_API_KEY", "AI21")
        ai21 = _import_optional("ai21", "ai21")
        chat_models = _import_optional("ai21.models.chat", "ai21")
        self.client = ai21.AI21Client(api_key=self._API_KEY)
        self.ChatMessage = chat_models.ChatMessage
        self.ResponseFormat = chat_models.ResponseFormat

    def _request_format(self, context, genparams):
        return {
            "prompt": context,
            "maxTokens": genparams.max_tokens,
            "temperature": genparams.temperature,
            "stopSequences": genparams.stop,
            "model": self._model,
        }

    def _response_format(self, response):
        return {"raw": response, "completion": response.choices[0].message.content}

    def _get_io_tokens(self, context, completion):
        return (
            completion["raw"].usage.prompt_tokens,
            completion["raw"].usage.completion_tokens,
        )

    def _api_call(self, endpoint, data, api_key, retries=3, retry_grace_time=2):
        last_error = None
        for attempt in range(retries):
            try:
                return self.client.chat.completions.create(
                    model=data["model"],
                    messages=[
                        self.ChatMessage(role="user", content=data["prompt"]),
                    ],
                    documents=[],
                    tools=[],
                    n=1,
                    max_tokens=data["maxTokens"],
                    temperature=data["temperature"],
                    top_p=1,
                    stop=data["stopSequences"],
                    response_format=self.ResponseFormat(type="text"),
                )
            except Exception as exc:
                last_error = exc
                if attempt < retries - 1:
                    time.sleep(retry_grace_time * (2**attempt))
        raise TimeoutError(f"AI21 request failed after {retries} attempts") from last_error


class CohereAIModelProvider(APIModelProvider):
    _ENDPOINT = os.environ.get(
        "COHERE_STUDIO_ENDPOINT", "https://api.ai21.com/studio/v1/{engine}/complete"
    )
    _NAME = "cohere"

    def __init__(self, model):
        self._model = model
        self._API_KEY = _require_env("COHERE_STUDIO_API_KEY", "Cohere")

    def _api_call(self, endpoint, data, api_key, retries=3, retry_grace_time=2):
        cohere = _import_optional("cohere", "cohere")
        client = cohere.Client(api_key)
        last_error = None
        for attempt in range(retries):
            try:
                response = client.generate(
                    model=self._model,
                    prompt=data["prompt"],
                    max_tokens=data["max_tokens"],
                    temperature=data["temperature"],
                    k=data["k"],
                    p=data["p"],
                    frequency_penalty=data["frequency_penalty"],
                    presence_penalty=data["presence_penalty"],
                    stop_sequences=data["stop_sequences"],
                    return_likelihoods=data["return_likelihoods"],
                )
                self.response = response
                return response
            except Exception as exc:
                last_error = exc
                if attempt < retries - 1:
                    time.sleep(retry_grace_time * (2**attempt))
        raise TimeoutError(f"Cohere request failed after {retries} attempts") from last_error

    def _request_format(self, context, genparams):
        return {
            "prompt": context,
            "model": self._model,
            "max_tokens": genparams.max_tokens,
            "temperature": genparams.temperature,
            "k": 1,
            "num_generations": 1,
            "p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "stop_sequences": genparams.stop,
            "return_likelihoods": "ALL",
        }

    def _response_format(self, response):
        try:
            completion = response.generations[0].text
        except Exception:
            completion = ""
        return {"raw": response, "completion": completion}

    def _get_io_tokens(self, context, completion):
        return len(context) / 1000, len(completion.get("completion", "")) / 1000


class ForeFrontAIModelProvider(APIModelProvider):
    _NAME = "ffai"
    _ENDPOINT_MAP = {
        "QA": "https://shared-api.forefront.link/organization/nKKlZP3F37RN/codegen-16b-nl/completions/eGQdyiZlHIW4",
        "CodeGen": "https://shared-api.forefront.link/organization/nKKlZP3F37RN/codegen-16b-nl/completions/eGQdyiZlHIW4",
        "Pythia": "https://shared-api.forefront.link/organization/nKKlZP3F37RN/pythia-20b/completions/vanilla",
    }

    def __init__(self, model="QA"):
        self._model = model
        self._API_KEY = _require_env("FOREFRONT_API_KEY", "Forefront")

    def _request_format(self, context, genparams):
        if count_tokens(context) + genparams.max_tokens >= 2047:
            context = context[-2048 + genparams.max_tokens :]
        return {
            "text": context,
            "numResults": 1,
            "length": genparams.max_tokens,
            "topKReturn": 1,
            "temperature": genparams.temperature,
            "stop": ["\n"],
            "logprobs": 1,
            "echo": False,
        }

    def _response_format(self, response):
        return {"raw": response, "completion": response["result"][0]["completion"]}

    def _get_io_tokens(self, context, completion):
        return count_tokens(context), len(completion["raw"]["logprobs"]["tokens"])

    def _get_endpoint(self):
        return self._ENDPOINT_MAP[self._model]


class TextSynthModelProvider(APIModelProvider):
    _ENDPOINT = os.environ.get(
        "TEXTSYNTH_ENDPOINT", "https://api.textsynth.com/v1/engines/{engine}/completions"
    )
    _NAME = "textsynth"

    def __init__(self, model="QA"):
        self._model = model
        self._API_KEY = _require_env("TEXTSYNTH_API_SECRET_KEY", "TextSynth")

    def _request_format(self, context, genparams):
        return {
            "prompt": context,
            "max_tokens": genparams.max_tokens,
            "temperature": genparams.temperature,
            "stop": genparams.stop,
        }

    def _response_format(self, response):
        return {"raw": response, "completion": response["text"]}

    def _get_io_tokens(self, context, completion):
        return completion["raw"]["input_tokens"], completion["raw"]["output_tokens"]


class AnthropicModelProvider(APIModelProvider):
    _ENDPOINT = os.environ.get("ANTHROPIC_ENDPOINT", "https://api.anthropic.com/v1/complete")
    _NAME = "anthropic"

    def __init__(self, model):
        self._model = model
        self._API_KEY = _require_env("ANTHROPIC_API_KEY", "Anthropic")
        anthropic = _import_optional("anthropic", "anthropic")
        self.client = anthropic.Anthropic(api_key=self._API_KEY)

    def _request_format(self, context, genparams):
        return {
            "prompt": context,
            "max_tokens_to_sample": genparams.max_tokens,
            "temperature": genparams.temperature,
            "model": self._model,
        }

    def _response_format(self, response):
        return {"raw": response, "completion": response.content[0].text}

    def _get_io_tokens(self, context, completion):
        usage = completion["raw"].usage
        return usage.input_tokens, usage.output_tokens

    def _api_call(self, endpoint, data, api_key, retries=3, retry_grace_time=2):
        last_error = None
        for attempt in range(retries):
            try:
                return self.client.messages.create(
                    model=data["model"],
                    max_tokens=data["max_tokens_to_sample"],
                    temperature=data["temperature"],
                    system="Follow the examples to only generate the answer. Do not generate any other texts.",
                    messages=[
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": f"{data['prompt']}"}],
                        }
                    ],
                )
            except Exception as exc:
                last_error = exc
                if attempt < retries - 1:
                    time.sleep(retry_grace_time * (2**attempt))
        raise TimeoutError(f"Anthropic request failed after {retries} attempts") from last_error


class GoogleModelProvider(APIModelProvider):
    _ENDPOINT = os.environ.get("GEMINI_ENDPOINT", "https://generativelanguage.googleapis.com")
    _NAME = "google"

    def __init__(self, model):
        self._model = model
        self._API_KEY = _require_env("GEMINI_API_KEY", "Gemini")
        self.genai = _import_optional("google.generativeai", "google")
        self.genai.configure(api_key=self._API_KEY)

    def _request_format(self, context, genparams):
        return {
            "prompt": context,
            "max_tokens_to_sample": genparams.max_tokens,
            "temperature": genparams.temperature,
            "model": self._model,
        }

    def _response_format(self, response):
        return {
            "raw": response,
            "completion": response.candidates[0].content.parts[0].text,
        }

    def _get_io_tokens(self, context, completion):
        usage = completion["raw"].usage_metadata
        return usage.prompt_token_count, usage.candidates_token_count

    def _api_call(self, endpoint, data, api_key, retries=3, retry_grace_time=2):
        last_error = None
        for attempt in range(retries):
            try:
                generation_config = {
                    "temperature": data["temperature"],
                    "top_p": 1,
                    "top_k": 1,
                    "max_output_tokens": data["max_tokens_to_sample"],
                    "response_mime_type": "text/plain",
                }
                model = self.genai.GenerativeModel(
                    model_name=data["model"],
                    generation_config=generation_config,
                )
                return model.start_chat(history=[]).send_message(data["prompt"])
            except Exception as exc:
                last_error = exc
                if attempt < retries - 1:
                    time.sleep(retry_grace_time * (2**attempt))
        raise TimeoutError(f"Gemini request failed after {retries} attempts") from last_error


class TogetherAIModelProvider(APIModelProvider):
    _ENDPOINT = os.environ.get("TOGETHERAI_ENDPOINT", "https://api.together.ai/v1/chat/completions")
    _NAME = "togetherai"

    def __init__(self, model):
        self._model = model
        self._API_KEY = _require_env("TOGETHER_API_KEY", "Together AI")

    def _request_format(self, context, genparams):
        return {
            "model": self._model,
            "messages": [{"content": context, "role": "user"}],
            "max_tokens": genparams.max_tokens,
            "temperature": genparams.temperature,
            "top_p": 0.7,
            "top_k": 50,
            "repetition_penalty": 1,
            "stop": genparams.stop,
            "stream": False,
        }

    def _response_format(self, response):
        return {
            "raw": response,
            "completion": response["choices"][0]["message"]["content"],
        }

    def _get_io_tokens(self, context, completion):
        usage = completion["raw"].get("usage", {})
        return usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)


class FakeModelProvider(ModelService):
    """Deterministic local provider for tests and business workflow demos."""

    _COSTS = {
        "support-cheap": {"input": 0.00000005, "output": 0.00000005},
        "support-strong": {"input": 0.0000005, "output": 0.0000005},
    }

    def __init__(self, model):
        self._model = model

    def getcompletion(
        self,
        context,
        use_save=False,
        savepath="raw.pkl",
        genparams=GenerationParameter(),
    ):
        started = time.time()
        payload = self._triage_payload(context)
        completion = json.dumps(payload, sort_keys=True)
        costs = self._COSTS.get(self._model, self._COSTS["support-cheap"])
        cost = count_tokens(context) * costs["input"] + count_tokens(completion) * costs["output"]
        return {
            "raw": payload,
            "completion": completion,
            "cost": cost,
            "latency": time.time() - started,
        }

    def _triage_payload(self, context):
        text = str(context).lower()
        strong = "strong" in self._model
        if any(term in text for term in ["refund", "charge", "invoice", "billing"]):
            category = "billing"
        elif any(term in text for term in ["demo", "pricing", "contract", "quote"]):
            category = "sales"
        elif any(term in text for term in ["api", "error", "down", "outage", "integration", "login"]):
            category = "technical"
        else:
            category = "general"

        urgent_terms = ["down", "outage", "security", "breach", "legal", "cannot login"]
        urgency = "high" if any(term in text for term in urgent_terms) else "normal"
        escalation = urgency == "high" or "angry" in text or "cancel" in text

        if strong:
            confidence = 0.93 if not escalation else 0.88
            summary = f"Strong triage: {category} request with {urgency} urgency."
            next_action = "route to specialist with context and draft a response"
        else:
            hard_case = escalation or len(text.split()) > 35 or "unclear" in text
            confidence = 0.56 if hard_case else 0.82
            summary = f"Cheap triage: likely {category} request."
            next_action = "answer from standard playbook" if confidence >= 0.7 else "escalate for stronger review"

        return {
            "category": category,
            "urgency": urgency,
            "summary": summary,
            "next_action": next_action,
            "confidence": confidence,
            "escalation": escalation,
        }


_PROVIDER_MAP = {
    "openai": OpenAIModelProvider,
    "ai21": AI21ModelProvider,
    "cohere": CohereAIModelProvider,
    "forefrontai": ForeFrontAIModelProvider,
    "textsynth": TextSynthModelProvider,
    "openaichat": OpenAIChatModelProvider,
    "anthropic": AnthropicModelProvider,
    "togetherai": TogetherAIModelProvider,
    "google": GoogleModelProvider,
    "fake": FakeModelProvider,
}


def make_model(provider, model):
    if provider not in _PROVIDER_MAP:
        raise ValueError(f"No model provider {provider!r} implemented")
    return _PROVIDER_MAP[provider](model)

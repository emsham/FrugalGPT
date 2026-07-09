import concurrent.futures
import threading
from collections import OrderedDict
from pathlib import Path

import pandas
from tqdm import tqdm

from service.modelservice import GenerationParameter, count_tokens, make_model
from .config import load_json_config, load_service_info
from .utils import getservicename


def form_keys(service_id, genparams, query):
    key_data = genparams.get_dict()
    key_data["service_id"] = service_id
    key_data["query"] = query
    return repr(key_data)


class LLMVanilla(object):
    def __init__(
        self,
        service_name=None,
        db_path="db/qa_cache.sqlite",
        db_path_new="db/HEADLINES.sqlite",
        max_workers=2,
        cache_enabled=True,
        service_info=None,
    ):
        self.max_workers = max_workers
        self.service_info = service_info or load_service_info()
        self.serviceidmap = load_json_config("serviceidmap.json")
        self.cache_enabled = cache_enabled
        self._cache_lock = threading.RLock()
        self._state_lock = threading.RLock()
        self.cost = 0
        self.latency = "NA"
        self.completion = None

        if service_name is None:
            service_name = getservicename()
        self.service_names = list(service_name)
        self.services = {}
        self._service_locks = {}
        for item in self.service_names:
            provider, name = item.split("/", 1)
            self.services[item] = make_model(provider, name)
            self._service_locks[item] = threading.RLock()

        self.cache = self._open_cache(db_path) if cache_enabled else {}
        self.cache_new = self._open_cache(db_path_new) if cache_enabled else {}

    def _open_cache(self, db_path):
        try:
            from sqlitedict import SqliteDict
        except ImportError as exc:
            raise RuntimeError(
                "sqlitedict is required for FrugalGPT caching. "
                "Install it or pass cache_enabled=False."
            ) from exc

        path = Path(db_path)
        if path.parent != Path("."):
            path.parent.mkdir(parents=True, exist_ok=True)
        return SqliteDict(str(path), autocommit=True)

    def get_cost(self):
        return self.cost

    def compute_cost(self, input_text, output_text, service_name):
        provider, model = service_name.split("/", 1)
        try:
            service_info = self.service_info[provider][model]
        except KeyError as exc:
            raise KeyError(f"No service cost config for {service_name}") from exc

        input_size = count_tokens(input_text)
        gen_size = count_tokens(output_text)
        cost = service_info["cost_input"] * input_size + service_info["cost_fixed"]
        if gen_size > service_info["fixed_size"]:
            cost += service_info["cost_output"] * (gen_size - service_info["fixed_size"])
        return cost

    def costestimate(
        self,
        query: str,
        service_name="fake/support-cheap",
        Params=GenerationParameter(max_tokens=50, temperature=0.1, stop=["\n"]),
    ):
        return self.compute_cost(
            input_text=query,
            output_text="token " * Params.max_tokens,
            service_name=service_name,
        )

    def _cache_key(self, service_name, genparams, query):
        service_id = self.serviceidmap.get(service_name, service_name)
        return form_keys(service_id=service_id, genparams=genparams, query=query)

    def _cache_get(self, key):
        if not self.cache_enabled:
            return None
        with self._cache_lock:
            value = self.cache.get(key)
            if value is not None and "cost" in value:
                self.cache_new[key] = value
                return value
        return None

    def _cache_set(self, key, value):
        if not self.cache_enabled:
            return
        with self._cache_lock:
            self.cache[key] = value

    def _get_completion_record(
        self,
        query: str,
        service_name="fake/support-cheap",
        use_save=False,
        use_db=True,
        savepath="raw.pkl",
        genparams=GenerationParameter(max_tokens=50, temperature=0.1, stop=["\n"]),
    ):
        model = self.services[service_name]
        key = self._cache_key(service_name, genparams, query)
        completion = self._cache_get(key) if use_db else None

        if completion is None:
            with self._service_locks[service_name]:
                completion = model.getcompletion(
                    query,
                    use_save=use_save,
                    savepath=savepath,
                    genparams=genparams,
                )
            if use_db:
                self._cache_set(key, completion)

        try:
            cost = completion["cost"]
        except KeyError:
            cost = self.compute_cost(
                input_text=query,
                output_text=completion["completion"],
                service_name=service_name,
            )
        latency = completion.get("latency", "NA")
        return {
            "answer": completion["completion"],
            "cost": cost,
            "latency": latency,
            "completion": completion,
        }

    def get_completion(
        self,
        query: str,
        service_name="fake/support-cheap",
        use_save=False,
        use_db=True,
        savepath="raw.pkl",
        genparams=GenerationParameter(max_tokens=50, temperature=0.1, stop=["\n"]),
        migrate_db=False,
    ):
        record = self._get_completion_record(
            query=query,
            service_name=service_name,
            use_save=use_save,
            use_db=use_db,
            savepath=savepath,
            genparams=genparams,
        )
        with self._state_lock:
            self.cost = record["cost"]
            self.completion = record["completion"]
            self.latency = record["latency"]
        return record["answer"]

    def get_completion_allservice(
        self,
        query: str,
        service_names=None,
        use_save=False,
        use_db=True,
        savepath="raw.pkl",
        genparams=GenerationParameter(max_tokens=50, temperature=0.1, stop=["\n"]),
    ):
        if service_names is None:
            service_names = []
        result = []
        for name in service_names:
            record = self._get_completion_record(
                query=query,
                service_name=name,
                use_save=use_save,
                use_db=use_db,
                savepath=savepath,
                genparams=genparams,
            )
            result.append({"service": name, "answer": record["answer"], "cost": record["cost"]})
        return pandas.DataFrame(result)

    def get_completion_batch(
        self,
        queries,
        service_name="fake/support-cheap",
        use_save=False,
        use_db=True,
        savepath="raw.pkl",
        genparams=GenerationParameter(max_tokens=50, temperature=0.1, stop=["\n"]),
    ):
        result = self.parallel_process_queries(
            queries, service_name, use_save, use_db, savepath, genparams
        )
        return pandas.DataFrame(result)

    def process_query(self, query, service_name, use_save, use_db, savepath, genparams):
        record = self._get_completion_record(
            query=query[0],
            service_name=service_name,
            use_save=use_save,
            use_db=use_db,
            savepath=savepath,
            genparams=genparams,
        )
        return {
            "_id": query[2],
            "answer": record["answer"],
            "ref_answer": query[1],
            "cost": record["cost"],
        }

    def parallel_process_queries(self, queries, service_name, use_save, use_db, savepath, genparams):
        result = OrderedDict()

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_query = {
                executor.submit(
                    self.process_query,
                    query,
                    service_name,
                    use_save,
                    use_db,
                    savepath,
                    genparams,
                ): query
                for query in queries
            }

            for future in tqdm(concurrent.futures.as_completed(future_to_query), total=len(future_to_query)):
                query = future_to_query[future]
                try:
                    data = future.result()
                    result[query[2]] = data
                except Exception as exc:
                    print(f"Query {query} generated an exception: {exc}")

        return list(result.values())

    def get_last_cost(self):
        return self.completion["cost"]

    def get_latency(self):
        return self.latency

    def reset(self):
        with self._state_lock:
            self.cost = 0
            self.completion = None

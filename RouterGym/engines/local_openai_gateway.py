"""Route logical RouterGym model keys to local OpenAI-compatible upstreams."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from RouterGym.engines.model_registry import ALL_MODELS, ModelEntry
from RouterGym.engines.openai_compatible import normalize_openai_compatible_base_url


DEFAULT_GATEWAY_HOST = "127.0.0.1"
DEFAULT_GATEWAY_PORT = 8000
ROUTE_ORDER: tuple[str, ...] = ("slm1", "slm2", "llm1", "llm2")


class GatewayConfigError(RuntimeError):
    """Raised when the local gateway route table is incomplete."""


@dataclass(frozen=True)
class GatewayRoute:
    """Mapping between a RouterGym logical key and one or more local upstreams."""

    model_key: str
    upstream_model_id: str
    upstream_base_urls: tuple[str, ...]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "model_key": self.model_key,
            "upstream_model_id": self.upstream_model_id,
            "upstream_base_urls": list(self.upstream_base_urls),
        }


def get_gateway_bind_host() -> str:
    return str(os.getenv("ROUTERGYM_GATEWAY_BIND_HOST") or DEFAULT_GATEWAY_HOST)


def get_gateway_bind_port() -> int:
    raw = str(os.getenv("ROUTERGYM_GATEWAY_BIND_PORT") or DEFAULT_GATEWAY_PORT).strip()
    try:
        return int(raw)
    except ValueError as exc:  # pragma: no cover - defensive only
        raise GatewayConfigError(f"Invalid ROUTERGYM_GATEWAY_BIND_PORT value: {raw!r}") from exc


def get_gateway_base_url() -> str:
    return normalize_openai_compatible_base_url(
        f"http://{get_gateway_bind_host()}:{get_gateway_bind_port()}"
    )


def _normalize_url_list(urls: Iterable[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    for raw in urls:
        value = str(raw or "").strip()
        if not value:
            continue
        normalized.append(normalize_openai_compatible_base_url(value))
    return tuple(normalized)


def _required_upstream_var_name(model_key: str) -> str:
    return f"ROUTERGYM_GATEWAY_{model_key.upper()}_UPSTREAM_BASE_URL"


def _replica_var_name(model_key: str) -> str:
    return f"ROUTERGYM_GATEWAY_{model_key.upper()}_REPLICA_BASE_URLS"


def _read_route(model_key: str, entry: ModelEntry) -> GatewayRoute:
    primary_name = _required_upstream_var_name(model_key)
    replica_name = _replica_var_name(model_key)
    primary_value = os.getenv(primary_name, "")
    replica_values = [
        value.strip()
        for value in str(os.getenv(replica_name, "")).split(",")
        if value.strip()
    ]
    upstream_base_urls = _normalize_url_list([primary_value, *replica_values])
    if not upstream_base_urls:
        raise GatewayConfigError(
            f"Missing local gateway route for {model_key}. "
            f"Set {primary_name} (and optionally {replica_name})."
        )
    return GatewayRoute(
        model_key=model_key,
        upstream_model_id=entry.hf_id,
        upstream_base_urls=upstream_base_urls,
    )


def build_gateway_routes(model_keys: Sequence[str] | None = None) -> Dict[str, GatewayRoute]:
    """Build the logical-key route table from environment variables."""

    keys = tuple(model_keys or ROUTE_ORDER)
    routes: Dict[str, GatewayRoute] = {}
    for model_key in keys:
        entry = ALL_MODELS.get(model_key)
        if entry is None:
            raise GatewayConfigError(f"Unknown model key: {model_key}")
        routes[model_key] = _read_route(model_key, entry)
    return routes


def choose_upstream_base_url(route: GatewayRoute, attempt_index: int = 0) -> str:
    """Pick an upstream URL, round-robin across replicas when present."""

    if not route.upstream_base_urls:
        raise GatewayConfigError(f"Route {route.model_key} has no upstream URLs configured.")
    return route.upstream_base_urls[attempt_index % len(route.upstream_base_urls)]


def build_gateway_models_payload(routes: Mapping[str, GatewayRoute]) -> Dict[str, Any]:
    """Return an OpenAI-style models payload exposing RouterGym logical keys."""

    data: List[Dict[str, Any]] = []
    for model_key in ROUTE_ORDER:
        route = routes.get(model_key)
        if route is None:
            continue
        data.append(
            {
                "id": route.model_key,
                "object": "model",
                "owned_by": "routergym-local-gateway",
                "permission": [],
                "routergym_upstream_model_id": route.upstream_model_id,
                "routergym_upstream_base_urls": list(route.upstream_base_urls),
            }
        )
    return {
        "object": "list",
        "data": data,
    }


__all__ = [
    "DEFAULT_GATEWAY_HOST",
    "DEFAULT_GATEWAY_PORT",
    "GatewayConfigError",
    "GatewayRoute",
    "ROUTE_ORDER",
    "build_gateway_models_payload",
    "build_gateway_routes",
    "choose_upstream_base_url",
    "get_gateway_base_url",
    "get_gateway_bind_host",
    "get_gateway_bind_port",
]

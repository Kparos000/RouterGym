"""Expose a single RouterGym-facing OpenAI-compatible gateway for four local models."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import traceback
from typing import Any, Dict, Mapping
from urllib import error as urlerror
from urllib import request

from RouterGym.engines.local_openai_gateway import (
    GatewayConfigError,
    GatewayRoute,
    build_gateway_models_payload,
    build_gateway_routes,
    choose_upstream_base_url,
    get_gateway_bind_host,
    get_gateway_bind_port,
)


class _GatewayServer(ThreadingHTTPServer):
    routes: Mapping[str, GatewayRoute]
    upstream_attempts: Dict[str, int]


class GatewayRequestHandler(BaseHTTPRequestHandler):
    server: _GatewayServer

    def log_message(self, format: str, *args: Any) -> None:  # pragma: no cover - console only
        super().log_message(format, *args)

    def _write_json(self, status: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") != "/v1/models":
            self._write_json(HTTPStatus.NOT_FOUND, {"error": {"message": "Not found"}})
            return
        self._write_json(HTTPStatus.OK, build_gateway_models_payload(self.server.routes))

    def do_POST(self) -> None:  # noqa: N802
        if self.path.rstrip("/") != "/v1/chat/completions":
            self._write_json(HTTPStatus.NOT_FOUND, {"error": {"message": "Not found"}})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length).decode("utf-8")
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise ValueError("Request body must be a JSON object.")
            requested_model = str(payload.get("model") or "").strip()
            route = self.server.routes.get(requested_model)
            if route is None:
                self._write_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": {"message": f"Unknown RouterGym model key: {requested_model!r}"}},
                )
                return
            attempt_index = self.server.upstream_attempts[requested_model]
            self.server.upstream_attempts[requested_model] += 1
            upstream_base_url = choose_upstream_base_url(route, attempt_index=attempt_index)
            forwarded_payload = dict(payload)
            forwarded_payload["model"] = route.upstream_model_id
            forwarded_body = json.dumps(forwarded_payload).encode("utf-8")
            upstream_endpoint = f"{upstream_base_url}/chat/completions"
            upstream_headers = {
                "Content-Type": "application/json",
                "Authorization": self.headers.get("Authorization", "Bearer EMPTY"),
            }
            req = request.Request(
                upstream_endpoint,
                data=forwarded_body,
                headers=upstream_headers,
                method="POST",
            )
            with request.urlopen(req, timeout=300) as response:
                upstream_raw = response.read()
                upstream_status = response.status
                upstream_content_type = response.headers.get("Content-Type", "application/json")
            self.send_response(upstream_status)
            self.send_header("Content-Type", upstream_content_type)
            self.send_header("Content-Length", str(len(upstream_raw)))
            self.end_headers()
            self.wfile.write(upstream_raw)
        except urlerror.HTTPError as exc:
            payload = exc.read().decode("utf-8", errors="replace")
            try:
                parsed = json.loads(payload)
            except Exception:
                parsed = {"error": {"message": payload}}
            self._write_json(exc.code, parsed)
        except Exception as exc:
            self._write_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {
                    "error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "stack_trace": traceback.format_exc(),
                    }
                },
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the RouterGym local OpenAI-compatible gateway."
    )
    parser.add_argument(
        "--host", type=str, default=get_gateway_bind_host(), help="Gateway bind host."
    )
    parser.add_argument(
        "--port", type=int, default=get_gateway_bind_port(), help="Gateway bind port."
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    routes = build_gateway_routes()
    server = _GatewayServer((args.host, args.port), GatewayRequestHandler)
    server.routes = routes
    server.upstream_attempts = defaultdict(int)
    print(
        json.dumps(
            {
                "status": "starting",
                "host": args.host,
                "port": args.port,
                "routes": {key: route.as_dict() for key, route in routes.items()},
            },
            indent=2,
        )
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:  # pragma: no cover - interactive stop path
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    try:
        main()
    except GatewayConfigError as exc:
        raise SystemExit(str(exc))

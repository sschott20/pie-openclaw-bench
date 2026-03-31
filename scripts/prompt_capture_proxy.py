#!/usr/bin/env python3
"""Logging proxy that captures Claude Code API requests.

Intercepts all requests to the Anthropic API, logs the full prompt
(system, tools, messages) to captured_prompts/, and forwards to the real API.

Usage:
    # Terminal 1: Start proxy
    python scripts/prompt_capture_proxy.py

    # Terminal 2: Run Claude Code through proxy
    ANTHROPIC_BASE_URL=http://localhost:5001 claude

    # Captured prompts saved to captured_prompts/
"""

import json
import time
import sys
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
import ssl
import urllib.request
import urllib.error

ANTHROPIC_API = "https://api.anthropic.com"
LISTEN_PORT = 5001
LOG_DIR = Path("captured_prompts")
LOG_DIR.mkdir(exist_ok=True)

request_counter = 0


class ProxyHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        global request_counter
        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len)

        # Log the request
        request_counter += 1
        ts = int(time.time())
        try:
            parsed = json.loads(body)

            # Extract module breakdown
            summary = self._summarize_request(parsed)

            log_entry = {
                "request_id": request_counter,
                "timestamp": ts,
                "path": self.path,
                "model": parsed.get("model", "unknown"),
                "summary": summary,
                "full_request": parsed,
            }

            log_path = LOG_DIR / f"{ts}_{request_counter:04d}.json"
            log_path.write_text(json.dumps(log_entry, indent=2, ensure_ascii=False))

            # Print summary to console
            print(f"\n{'='*60}")
            print(f"Request #{request_counter} → {self.path}")
            print(f"  Model: {parsed.get('model', '?')}")
            if summary.get("system_tokens"):
                print(f"  System prompt: ~{summary['system_tokens']} tokens")
            if summary.get("num_tools"):
                print(f"  Tools: {summary['num_tools']} ({summary.get('tool_tokens', '?')} tokens)")
            if summary.get("num_messages"):
                print(f"  Messages: {summary['num_messages']}")
            if summary.get("total_est_tokens"):
                print(f"  Total est tokens: ~{summary['total_est_tokens']}")
            print(f"  Saved to: {log_path.name}")
            print(f"{'='*60}")

        except json.JSONDecodeError:
            log_path = LOG_DIR / f"{ts}_{request_counter:04d}_raw.bin"
            log_path.write_bytes(body)
            print(f"  [non-JSON request saved to {log_path.name}]")

        # Forward to real API
        url = f"{ANTHROPIC_API}{self.path}"
        headers = {}
        for key, val in self.headers.items():
            if key.lower() not in ("host", "content-length", "transfer-encoding"):
                headers[key] = val

        req = urllib.request.Request(url, data=body, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req) as resp:
                resp_body = resp.read()
                self.send_response(resp.status)
                for key, val in resp.getheaders():
                    if key.lower() not in ("transfer-encoding", "content-encoding", "content-length"):
                        self.send_header(key, val)
                self.send_header("Content-Length", str(len(resp_body)))
                self.end_headers()
                self.wfile.write(resp_body)
        except urllib.error.HTTPError as e:
            resp_body = e.read()
            self.send_response(e.code)
            for key, val in e.headers.items():
                if key.lower() not in ("transfer-encoding", "content-encoding", "content-length"):
                    self.send_header(key, val)
            self.send_header("Content-Length", str(len(resp_body)))
            self.end_headers()
            self.wfile.write(resp_body)

    def do_GET(self):
        # Forward GET requests (health checks, etc.)
        url = f"{ANTHROPIC_API}{self.path}"
        headers = {k: v for k, v in self.headers.items() if k.lower() != "host"}
        req = urllib.request.Request(url, headers=headers, method="GET")
        try:
            with urllib.request.urlopen(req) as resp:
                resp_body = resp.read()
                self.send_response(resp.status)
                for key, val in resp.getheaders():
                    if key.lower() not in ("transfer-encoding", "content-encoding", "content-length"):
                        self.send_header(key, val)
                self.send_header("Content-Length", str(len(resp_body)))
                self.end_headers()
                self.wfile.write(resp_body)
        except urllib.error.HTTPError as e:
            self.send_response(e.code)
            self.end_headers()

    def log_message(self, format, *args):
        # Suppress default access logs
        pass

    def _summarize_request(self, parsed: dict) -> dict:
        """Extract module breakdown from the request."""
        summary = {}

        # System prompt
        system = parsed.get("system")
        if system:
            if isinstance(system, str):
                summary["system_tokens"] = len(system) // 4
                summary["system_chars"] = len(system)
            elif isinstance(system, list):
                total = sum(len(s.get("text", "")) for s in system if isinstance(s, dict))
                summary["system_tokens"] = total // 4
                summary["system_chars"] = total
                summary["system_blocks"] = len(system)

        # Tools
        tools = parsed.get("tools", [])
        if tools:
            summary["num_tools"] = len(tools)
            tool_json = json.dumps(tools)
            summary["tool_tokens"] = len(tool_json) // 4
            summary["tool_names"] = [t.get("name", "?") for t in tools]

        # Messages
        messages = parsed.get("messages", [])
        summary["num_messages"] = len(messages)

        total_msg_chars = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total_msg_chars += len(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        total_msg_chars += len(block.get("text", ""))
        summary["message_tokens"] = total_msg_chars // 4

        # Total estimate
        summary["total_est_tokens"] = (
            summary.get("system_tokens", 0)
            + summary.get("tool_tokens", 0)
            + summary.get("message_tokens", 0)
        )

        return summary


def main():
    print(f"Prompt Capture Proxy")
    print(f"  Listening on: http://localhost:{LISTEN_PORT}")
    print(f"  Forwarding to: {ANTHROPIC_API}")
    print(f"  Saving to: {LOG_DIR.absolute()}")
    print(f"\nTo use with Claude Code:")
    print(f"  ANTHROPIC_BASE_URL=http://localhost:{LISTEN_PORT} claude")
    print(f"\nWaiting for requests...\n")

    server = HTTPServer(("localhost", LISTEN_PORT), ProxyHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print(f"\n\nStopped. Captured {request_counter} requests to {LOG_DIR}/")
        server.server_close()


if __name__ == "__main__":
    main()

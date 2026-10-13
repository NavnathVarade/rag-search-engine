"""A local STAND-IN for an OpenAI-compatible /v1/chat/completions endpoint, used only by the e2e tests.

It is not a language model: it reads the numbered <context id="N"> blocks the real answerer put in the prompt and
replies with fixed Markdown that cites the first one. That exercises the real provider, prompt building, citation
validation and the UI's Markdown/citation rendering. It says nothing about the quality of real model answers."""
import json
import re
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        prompt = body["messages"][-1]["content"]
        ids = re.findall(r'<context id="(\d+)"', prompt)
        if ids:
            text = (
                "A failed container is restarted by the **kubelet** according to the pod's `restartPolicy` [1].\n\n"
                "- Allowed policies: Always, OnFailure, Never [1]\n- Repeated failures lead to *CrashLoopBackOff* [1]"
            )
        else:
            text = "I don't have enough evidence in the indexed documentation to answer that reliably."
        out = {"model": "stand-in-model", "choices": [{"message": {"content": text}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 120, "completion_tokens": 40}}
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    HTTPServer(("127.0.0.1", int(sys.argv[1]) if len(sys.argv) > 1 else 9100), Handler).serve_forever()

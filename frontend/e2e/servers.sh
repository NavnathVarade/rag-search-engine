#!/usr/bin/env bash
# Start (or restart) the two REAL API servers the Playwright tests run against, from the repo root:
#   :8010  AUTH_ENABLED=false  (open dev server, serves frontend/dist)
#   :8011  AUTH_ENABLED=true   (API keys e2e-admin-key / e2e-user-key, METRICS_TOKEN=m3trics)
#   :8012  LLM configured, pointing at a local stand-in for an OpenAI-compatible API on :9100
# Usage: frontend/e2e/servers.sh start|stop
set -euo pipefail
cd "$(dirname "$0")/../.."
export NO_PROXY=127.0.0.1,localhost
stop() { for p in 8010 8011 8012 9100; do pid=$(lsof -t -i:$p 2>/dev/null || true); [ -n "$pid" ] && kill $pid || true; done; sleep 1; }
case "${1:-start}" in
  stop) stop ;;
  start)
    stop
    H1=$(printf %s e2e-admin-key | sha256sum | cut -d' ' -f1)
    H2=$(printf %s e2e-user-key | sha256sum | cut -d' ' -f1)
    AJ="{\"$H1\":{\"user_id\":\"admin1\",\"tenant_id\":\"t1\",\"groups\":[],\"admin\":true},\"$H2\":{\"user_id\":\"alice\",\"tenant_id\":\"t1\",\"groups\":[\"eng\"],\"admin\":false}}"
    AUTH_ENABLED=false nohup uv run --frozen uvicorn rag_engine.api.app:app --port 8010 --log-level warning > /tmp/e2e-open.log 2>&1 &
    AUTH_ENABLED=true API_KEYS_JSON="$AJ" METRICS_TOKEN=m3trics nohup uv run --frozen uvicorn rag_engine.api.app:app --port 8011 --log-level warning > /tmp/e2e-auth.log 2>&1 &
    # :8012 uses an OpenAI-compatible API served by e2e/fake_llm.py (a stand-in, not a real model) on :9100
    nohup uv run --frozen python frontend/e2e/fake_llm.py 9100 > /tmp/e2e-llm.log 2>&1 &
    AUTH_ENABLED=false LLM_PROVIDER=openai_compatible LLM_BASE_URL=http://127.0.0.1:9100/v1 LLM_API_KEY=test-key LLM_MODEL=stand-in-model nohup uv run --frozen uvicorn rag_engine.api.app:app --port 8012 --log-level warning > /tmp/e2e-llm-api.log 2>&1 &
    for i in $(seq 1 30); do
      curl -fs localhost:8010/health >/dev/null 2>&1 && curl -fs localhost:8011/health >/dev/null 2>&1 && curl -fs localhost:8012/health >/dev/null 2>&1 && { echo "servers up on :8010, :8011, :8012"; exit 0; }
      sleep 0.5
    done
    echo "servers failed to start; see /tmp/e2e-*.log" >&2; exit 1 ;;
esac

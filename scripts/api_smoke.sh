#!/usr/bin/env bash
# Smoke test of the public API against the spec (CLAUDE.md section 2).
# Usage: bash scripts/api_smoke.sh [URL] [KEY]
#   URL default: http://127.0.0.1:${PORT:-8000}   KEY default: API_KEY from .env
#   env: VIDEO=<path.mov> (default data/open/videos/X-1.mov)  POLL_TIMEOUT=300  POLL_EVERY=2
# Exit code 0 when every check passed.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [ -f "$ROOT/.env" ]; then set -a; . "$ROOT/.env"; set +a; fi
URL="${1:-http://127.0.0.1:${PORT:-8000}}"; URL="${URL%/}"
KEY="${2:-${API_KEY:-}}"
VIDEO="${VIDEO:-$ROOT/data/open/videos/X-1.mov}"
POLL_TIMEOUT="${POLL_TIMEOUT:-300}"
POLL_EVERY="${POLL_EVERY:-2}"
[ -n "$KEY" ] || { echo "no API key: pass it as the 2nd argument or set API_KEY in .env"; exit 2; }
[ -f "$VIDEO" ] || { echo "video not found: $VIDEO"; exit 2; }

PASS=0; FAIL=0
ok()  { echo "  PASS  $*"; PASS=$((PASS + 1)); }
bad() { echo "  FAIL  $*"; FAIL=$((FAIL + 1)); }
code() { curl -s -o /dev/null -w '%{http_code}' --max-time 120 "$@"; }
# jget <key>: print a top-level JSON field from stdin ("" when absent).
jget() { python3 -I -c 'import json,sys
d = json.load(sys.stdin); v = d.get(sys.argv[1]) if isinstance(d, dict) else None
print("" if v is None else (len(v) if isinstance(v, list) else v))' "$1" 2>/dev/null; }
expect() {  # expect <label> <want_code> <curl args...>
  local label="$1" want="$2"; shift 2
  local got; got="$(code "$@")"
  [ "$got" = "$want" ] && ok "$label -> $got" || bad "$label -> $got (want $want)"
}
# poll <job_id> <want_status>: wait until the job leaves `processing`; prints the body.
poll() {
  local job="$1" want="$2" t0 st body
  t0=$(date +%s)
  while :; do
    body="$(curl -s --max-time 30 "$URL/api/v1/jobs/$job" -H "X-API-Key: $KEY")"
    st="$(printf '%s' "$body" | jget status)"
    if [ "$st" != "processing" ] && [ -n "$st" ]; then break; fi
    if [ $(( $(date +%s) - t0 )) -ge "$POLL_TIMEOUT" ]; then st="timeout"; break; fi
    sleep "$POLL_EVERY"
  done
  local dt=$(( $(date +%s) - t0 ))
  [ "$st" = "$want" ] && ok "job $job -> $st after ${dt}s" || bad "job $job -> $st after ${dt}s (want $want)"
  printf '%s' "$body"
}

echo "== $URL"
echo "-- health (public)"
H="$(curl -s --max-time 30 "$URL/health")"; echo "  $H"
[ "$(printf '%s' "$H" | jget status)" = "ok" ] && ok "GET /health" || bad "GET /health"

echo "-- auth"
expect "GET jobs without key"   401 "$URL/api/v1/jobs/deadbeef"
expect "GET jobs wrong key"     401 -H "X-API-Key: wrong-key-$RANDOM" "$URL/api/v1/jobs/deadbeef"
expect "POST analyze no key"    401 -X POST "$URL/api/v1/analyze" -F "file=@$VIDEO"
expect "POST analyze wrong key" 401 -X POST -H "X-API-Key: wrong-key-$RANDOM" "$URL/api/v1/analyze" -F "file=@$VIDEO"

echo "-- bad requests"
expect "GET unknown job"        404 -H "X-API-Key: $KEY" "$URL/api/v1/jobs/deadbeef"
expect "GET malformed job id"   404 -H "X-API-Key: $KEY" "$URL/api/v1/jobs/..%2F..%2Fetc"
expect "POST without 'file'"    400 -X POST -H "X-API-Key: $KEY" "$URL/api/v1/analyze" -F "video=@$VIDEO"
expect "POST empty body"        400 -X POST -H "X-API-Key: $KEY" "$URL/api/v1/analyze"

echo "-- analyze $(basename "$VIDEO")"
T0=$(date +%s)
R="$(curl -s --max-time 120 -X POST "$URL/api/v1/analyze" -H "X-API-Key: $KEY" -F "file=@$VIDEO")"
echo "  $R"
JOB="$(printf '%s' "$R" | jget job_id)"
if [[ "$JOB" =~ ^[0-9a-f]{8}$ ]]; then
  ok "job_id=$JOB"
  BODY="$(poll "$JOB" done)"
  echo "  verdict=$(printf '%s' "$BODY" | jget verdict) violations=$(printf '%s' "$BODY" | jget violations) note=$(printf '%s' "$BODY" | jget note)"
  echo "  elapsed since upload: $(( $(date +%s) - T0 ))s; response bytes: ${#BODY}"
  printf '%s' "$BODY" | python3 -I -c 'import json,sys
d = json.load(sys.stdin); keys = list(d)[:4]
sys.exit(0 if keys == ["job_id", "status", "verdict", "violations"] else 1)' \
    && ok "field order job_id,status,verdict,violations" || bad "field order: $(printf '%s' "$BODY" | head -c 200)"
else
  bad "analyze did not return a job_id"
fi

echo "-- broken file"
BAD="$(mktemp "${TMPDIR:-/tmp}/bad.XXXXXX.mov")"; echo junk > "$BAD"
R="$(curl -s --max-time 60 -X POST "$URL/api/v1/analyze" -H "X-API-Key: $KEY" -F "file=@$BAD")"
rm -f "$BAD"
JOB="$(printf '%s' "$R" | jget job_id)"
if [[ "$JOB" =~ ^[0-9a-f]{8}$ ]]; then
  BODY="$(poll "$JOB" failed)"
  echo "  error=$(printf '%s' "$BODY" | jget error)"
else
  bad "broken-file upload did not return a job_id: $R"
fi

echo "== passed $PASS, failed $FAIL"
[ "$FAIL" -eq 0 ]

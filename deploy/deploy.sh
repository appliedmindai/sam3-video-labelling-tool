#!/usr/bin/env bash
#
# deploy.sh — provision, deploy, inspect, and tear down the SAM3 Video
# Labelling Tool on Google Cloud Run (L4 GPU) + GCS.
#
# Usage:
#   ./deploy/deploy.sh [command] [flags]
#
# Commands:
#   up                 (default) idempotent provision + deploy. Safe to
#                      re-run: every step checks current state and skips
#                      work that is already done.
#   status             show configured + live deployment state
#   logs [-f]          read recent Cloud Run logs (-f tails; needs gcloud beta)
#   password           print the current shared password (from service env)
#   generate-password  print a fresh three-word password (no GCP calls)
#   smoke              authenticated smoke checks against the deployed URL
#   down [--purge]     delete resources this script created (state-tracked)
#
# Flags:
#   --yes / -y         skip confirmation prompts. REQUIRED for `down` in a
#                      non-interactive shell.
#   --rotate-password  generate a new password on this `up`
#   --purge            with `down`: also delete the GCS buckets (annotation
#                      sessions + model weights) after an extra confirmation
#
# Configuration (env vars; only PROJECT is needed on the first run — every
# value is remembered in deploy/.deploy-state.json afterwards):
#   PROJECT          GCP project id (falls back to `gcloud config` default)
#   REGION           Cloud Run region          (default: us-east4 — has L4s)
#   SERVICE          Cloud Run service name    (default: sam3-annotator)
#   REPO_NAME        Artifact Registry repo    (default: cloud-run-images)
#   MODELS_BUCKET    SAM3 weights bucket       (default: $PROJECT-sam3-models)
#   SESSIONS_BUCKET  annotation data bucket    (default: $PROJECT-sam3-sessions)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
STATE_FILE="$SCRIPT_DIR/.deploy-state.json"

# ---------------------------------------------------------------- logging --

say()  { echo "[deploy] $*"; }
warn() { echo "[deploy] WARNING: $*" >&2; }
die()  { echo "[deploy] ERROR: $*" >&2; exit 1; }

# ------------------------------------------------------------ state file --
# .deploy-state.json is a flat string->string map. Keys:
#   project / region / service / repo_name / models_bucket / sessions_bucket
#     — remembered config so later commands need no env vars
#   created_registry / created_sa / created_service /
#   created_models_bucket / created_sessions_bucket / uploaded_weights
#     — "true" when THIS script created the resource. `down` only deletes
#       resources marked created; pre-existing ones are never touched.
#   last_image / last_deploy_at — informational, shown by `status`.

state_get() { # state_get KEY -> value (empty if missing)
  [ -f "$STATE_FILE" ] || { echo ""; return; }
  python3 - "$1" "$STATE_FILE" <<'PYEOF'
import json, sys
key, path = sys.argv[1], sys.argv[2]
try:
    data = json.load(open(path))
except Exception:
    data = {}
print(data.get(key, ""))
PYEOF
}

state_set() { # state_set KEY VALUE
  python3 - "$1" "$2" "$STATE_FILE" <<'PYEOF'
import json, os, sys
key, value, path = sys.argv[1], sys.argv[2], sys.argv[3]
data = {}
if os.path.exists(path):
    try:
        data = json.load(open(path))
    except Exception:
        data = {}
data[key] = value
tmp = path + ".tmp"
with open(tmp, "w") as f:
    json.dump(data, f, indent=2, sort_keys=True)
    f.write("\n")
os.replace(tmp, path)
PYEOF
}

# ------------------------------------------------------------ prompts -----
# Repo convention: prompts must be TTY-guarded. Non-interactive `up` runs
# auto-proceed with a log line (never a silent exit 0). Destructive
# commands (`down`) REFUSE to run non-interactively without --yes.

confirm() { # confirm "question"
  if [ "$YES" = "true" ]; then return 0; fi
  if [ -t 0 ]; then
    read -r -p "[deploy] $1 (y/N) " reply
    [[ "$reply" =~ ^[Yy]$ ]] || { say "Aborted."; exit 1; }
  else
    say "non-interactive shell — auto-proceeding (stdin is not a TTY)"
  fi
}

confirm_destructive() { # confirm_destructive "question"
  if [ "$YES" = "true" ]; then return 0; fi
  if [ -t 0 ]; then
    read -r -p "[deploy] $1 (y/N) " reply
    [[ "$reply" =~ ^[Yy]$ ]] || { say "Aborted."; exit 1; }
  else
    die "destructive command in a non-interactive shell requires --yes"
  fi
}

# Bucket purge deletes irrecoverable annotation data. Interactively we
# ALWAYS prompt, even with --yes; non-interactively --yes is required
# (enforced by confirm_destructive earlier) and we proceed loudly.
confirm_purge() {
  if [ -t 0 ]; then
    read -r -p "[deploy] $1 (y/N) " reply
    [[ "$reply" =~ ^[Yy]$ ]] || { say "Aborted."; exit 1; }
  else
    warn "non-interactive purge — deleting buckets because --yes and --purge were both given"
  fi
}

# ------------------------------------------------------------- config -----
# Resolution order: env var > state file > default. The first `up`
# persists everything, so day-2 commands are just `./deploy/deploy.sh logs`.

resolve_config() {
  PROJECT="${PROJECT:-$(state_get project)}"
  if [ -z "$PROJECT" ]; then
    PROJECT="$(gcloud config get-value project 2>/dev/null || true)"
    [ -n "$PROJECT" ] && warn "PROJECT not set — using gcloud default '$PROJECT'"
  fi
  [ -n "$PROJECT" ] || die "set PROJECT=<gcp-project-id> (first run only — it is then remembered in $STATE_FILE)"

  REGION="${REGION:-$(state_get region)}";                 REGION="${REGION:-us-east4}"
  SERVICE="${SERVICE:-$(state_get service)}";              SERVICE="${SERVICE:-sam3-annotator}"
  REPO_NAME="${REPO_NAME:-$(state_get repo_name)}";        REPO_NAME="${REPO_NAME:-cloud-run-images}"
  MODELS_BUCKET="${MODELS_BUCKET:-$(state_get models_bucket)}"
  MODELS_BUCKET="${MODELS_BUCKET:-$PROJECT-sam3-models}"
  SESSIONS_BUCKET="${SESSIONS_BUCKET:-$(state_get sessions_bucket)}"
  SESSIONS_BUCKET="${SESSIONS_BUCKET:-$PROJECT-sam3-sessions}"

  SA_NAME="sam3-annotator"
  SA_EMAIL="$SA_NAME@$PROJECT.iam.gserviceaccount.com"
  CACHE_IMAGE="$REGION-docker.pkg.dev/$PROJECT/$REPO_NAME/sam3-native:latest"
}

persist_config() {
  state_set project "$PROJECT"
  state_set region "$REGION"
  state_set service "$SERVICE"
  state_set repo_name "$REPO_NAME"
  state_set models_bucket "$MODELS_BUCKET"
  state_set sessions_bucket "$SESSIONS_BUCKET"
}

# ----------------------------------------------------------- password -----
# Three words from a ~250-word list ≈ 24 bits of entropy — right-sized for
# "keep strangers off my GPU", not bank-grade. secrets.choice is CSPRNG.

generate_password() {
  python3 - <<'PYEOF'
import secrets
WORDS = sorted(set("""
otter fox wolf bear hawk crane heron finch robin wren lynx hare moose bison seal whale
crab squid gecko viper cobra eagle falcon raven crow dove swan goose duck owl ibis stork
salmon trout perch bass pike carp shark manta orca walrus badger weasel marten stoat ferret mole
red blue green amber coral teal cyan navy plum rose gold silver copper bronze ivory pearl
crimson scarlet maroon violet indigo lilac mauve olive lime mint sage jade ruby topaz onyx opal
river lake pond creek brook delta marsh dune mesa butte cliff ridge peak summit valley glen
forest grove birch cedar maple oak pine aspen willow alder elm fir spruce laurel hazel rowan
storm cloud frost snow hail sleet mist fog dew rain wind gale breeze thunder ember spark
lantern anchor compass kettle hammer chisel ladder bucket barrel basket candle mirror saddle paddle anvil crate
button ribbon needle spool loom quill scroll ledger atlas globe prism lens gear pulley lever wheel
apple pear peach mango grape lemon melon berry fig date honey clove basil thyme ginger pepper
drum flute cello banjo chord tempo waltz polka opera lyric verse hymn organ viola reed bell
comet nova lunar solar orbit nebula quasar meteor zenith aurora eclipse cosmos saturn vega rigel astro
tundra steppe prairie canyon fjord atoll lagoon harbor jetty quay wharf isle reef shoal cove bay
baker mason smith potter weaver tailor scribe ranger pilot sailor scout knight squire bard monk friar
pebble boulder quartz flint slate marble granite basalt coal iron zinc cobalt nickel chrome cliffs dunes
""".split()))
print("-".join(secrets.choice(WORDS) for _ in range(3)))
PYEOF
}

# ---------------------------------------------------------- source hash ---
# The image tag is the hash of every tracked file that goes into the build.
# Same source → same tag → Cloud Build is skipped entirely. Working-tree
# content is hashed (uncommitted edits count); brand-new untracked files
# don't change the hash until `git add` (acceptable: real deploys flow
# through git).

compute_source_hash() {
  ( cd "$REPO_ROOT" &&
    git ls-files backend frontend Dockerfile.native .dockerignore \
        deploy/nginx.conf deploy/entrypoint.sh cloudbuild-native.yaml \
      | LC_ALL=C sort \
      | tr '\n' '\0' \
      | xargs -0 shasum -a 256 \
      | shasum -a 256 | cut -c1-12 )
}

# -------------------------------------------------------- live queries ----

service_exists() {
  gcloud run services describe "$SERVICE" --region "$REGION" \
    --project "$PROJECT" >/dev/null 2>&1
}

# Tri-state existence probe for ownership decisions: "exists", "absent",
# or die. Unlike service_exists, a transient gcloud failure must NOT be
# read as "absent" — that would mark a pre-existing service as
# created-by-us and let `down` delete it later.
service_probe() {
  local err
  if err=$(gcloud run services describe "$SERVICE" --region "$REGION" \
      --project "$PROJECT" --format='value(metadata.name)' 2>&1); then
    echo "exists"
  elif grep -qi "cannot find\|not[ _]*found\|could not be found\|does not exist" <<<"$err"; then
    echo "absent"
  else
    die "could not determine whether service $SERVICE exists (refusing to guess ownership): $err"
  fi
}

service_url() {
  gcloud run services describe "$SERVICE" --region "$REGION" \
    --project "$PROJECT" --format='value(status.url)' 2>/dev/null || true
}

deployed_image() {
  gcloud run services describe "$SERVICE" --region "$REGION" \
    --project "$PROJECT" \
    --format='value(spec.template.spec.containers[0].image)' 2>/dev/null || true
}

get_deployed_password() {
  gcloud run services describe "$SERVICE" --region "$REGION" \
    --project "$PROJECT" --format=json 2>/dev/null | python3 -c '
import json, sys
try:
    svc = json.load(sys.stdin)
    envs = svc["spec"]["template"]["spec"]["containers"][0].get("env", [])
    print(next((e.get("value", "") for e in envs if e.get("name") == "AUTH_PASSWORD"), ""))
except Exception:
    print("")
'
}

# ------------------------------------------------------------ up steps ----

step_preflight() {
  say "1/8 preflight"
  command -v gcloud >/dev/null || die "gcloud CLI not found — install the Google Cloud SDK"
  command -v gsutil >/dev/null || die "gsutil not found — install the Google Cloud SDK"
  command -v python3 >/dev/null || die "python3 not found"
  command -v git >/dev/null || die "git not found"

  local account
  account=$(gcloud auth list --filter=status:ACTIVE --format='value(account)' 2>/dev/null || true)
  [ -n "$account" ] || die "no active gcloud account — run: gcloud auth login"
  say "  account: $account"
  say "  project: $PROJECT  region: $REGION  service: $SERVICE"

  gcloud projects describe "$PROJECT" >/dev/null 2>&1 \
    || die "project '$PROJECT' not found or no access"

  # Billing must be on or API enablement / Cloud Build will fail with an
  # opaque error later. Check it now with a clear message instead.
  local billing
  billing=$(gcloud billing projects describe "$PROJECT" \
    --format='value(billingEnabled)' 2>/dev/null || echo "unknown")
  if [ "$billing" = "False" ]; then
    die "billing is not enabled on '$PROJECT' — link a billing account first"
  elif [ "$billing" = "unknown" ]; then
    warn "could not verify billing (missing permission?) — continuing"
  fi
}

step_enable_apis() {
  say "2/8 enable APIs"
  local needed="run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com storage.googleapis.com"
  local enabled
  enabled=$(gcloud services list --enabled --project "$PROJECT" \
    --format='value(config.name)')
  local api
  for api in $needed; do
    if grep -qx "$api" <<<"$enabled"; then
      say "  $api already enabled — skip"
    else
      say "  enabling $api"
      gcloud services enable "$api" --project "$PROJECT"
      # Note: `down` never disables APIs — other workloads may rely on them.
    fi
  done
}

step_registry() {
  say "3/8 Artifact Registry repo"
  if gcloud artifacts repositories describe "$REPO_NAME" --location "$REGION" \
       --project "$PROJECT" >/dev/null 2>&1; then
    say "  repo $REPO_NAME exists — skip"
  else
    say "  creating repo $REPO_NAME in $REGION"
    gcloud artifacts repositories create "$REPO_NAME" \
      --repository-format=docker --location "$REGION" --project "$PROJECT"
    state_set created_registry "true"
  fi
}

step_buckets() {
  say "4/8 GCS buckets"
  local name key
  for spec in "$MODELS_BUCKET:created_models_bucket" "$SESSIONS_BUCKET:created_sessions_bucket"; do
    name="${spec%%:*}"; key="${spec##*:}"
    if gsutil ls -b "gs://$name" >/dev/null 2>&1; then
      say "  gs://$name exists — skip"
    else
      say "  creating gs://$name in $REGION"
      gsutil mb -p "$PROJECT" -l "$REGION" "gs://$name"
      state_set "$key" "true"
    fi
  done
}

step_weights() {
  say "5/8 SAM3 model weights"
  if gsutil -q stat "gs://$MODELS_BUCKET/sam3/config.json" 2>/dev/null; then
    say "  weights present in gs://$MODELS_BUCKET/sam3/ — skip (never re-pushes ~7 GB)"
    return
  fi
  # The gated facebook/sam3 model needs a HuggingFace login — the one
  # genuinely manual prerequisite. Download once, upload once; after that
  # this step always skips.
  command -v hf >/dev/null || die "weights missing from gs://$MODELS_BUCKET/sam3/ and the 'hf' CLI is not installed.
  Fix:  pip install -U huggingface_hub
        hf auth login            # account needs access to gated facebook/sam3
        ./deploy/deploy.sh up    # re-run"
  local tmp
  tmp=$(mktemp -d)
  say "  downloading facebook/sam3 (~7 GB) to $tmp"
  hf download facebook/sam3 --local-dir "$tmp/sam3"
  say "  uploading to gs://$MODELS_BUCKET/sam3/"
  gsutil -m cp -r "$tmp/sam3/"* "gs://$MODELS_BUCKET/sam3/"
  rm -rf "$tmp"
  state_set uploaded_weights "true"
}

step_service_account() {
  say "6/8 service account"
  if gcloud iam service-accounts describe "$SA_EMAIL" --project "$PROJECT" >/dev/null 2>&1; then
    say "  $SA_EMAIL exists — skip"
  else
    say "  creating $SA_EMAIL"
    gcloud iam service-accounts create "$SA_NAME" \
      --display-name="SAM3 Video Labelling Tool" --project "$PROJECT"
    state_set created_sa "true"
  fi
  # Idempotent: re-granting an existing binding is a no-op.
  gsutil iam ch "serviceAccount:$SA_EMAIL:roles/storage.objectAdmin" \
    "gs://$SESSIONS_BUCKET" >/dev/null
  say "  $SA_EMAIL has storage.objectAdmin on gs://$SESSIONS_BUCKET"
}

step_build() {
  say "7/8 container image"
  SRC_HASH=$(compute_source_hash)
  IMAGE="$REGION-docker.pkg.dev/$PROJECT/$REPO_NAME/sam3-native:$SRC_HASH"
  if gcloud artifacts docker images describe "$IMAGE" --project "$PROJECT" >/dev/null 2>&1; then
    say "  image for source hash $SRC_HASH already in registry — skip Cloud Build"
    return
  fi
  say "  building $IMAGE (first build ~15-20 min; cached code-only builds ~3-6 min)"
  gcloud builds submit \
    --config="$REPO_ROOT/cloudbuild-native.yaml" \
    --substitutions=_IMAGE="$IMAGE",_CACHE_IMAGE="$CACHE_IMAGE",_MODELS_BUCKET="$MODELS_BUCKET" \
    --project "$PROJECT" \
    "$REPO_ROOT"
}

step_deploy() {
  say "8/8 deploy to Cloud Run"
  local current_image="" current_password="" probe
  probe=$(service_probe)   # dies on transient errors — never guesses ownership
  if [ "$probe" = "exists" ]; then
    current_image=$(deployed_image)
    current_password=$(get_deployed_password)
  fi

  # Reuse the live password on redeploys so users/browsers stay logged in;
  # generate fresh only on first deploy or explicit --rotate-password.
  if [ -n "$current_password" ] && [ "$ROTATE_PASSWORD" != "true" ]; then
    PASSWORD="$current_password"
  else
    PASSWORD=$(generate_password)
    [ -n "$current_password" ] && say "  rotating password (old one stops working)"
  fi

  if [ "$current_image" = "$IMAGE" ] && [ "$PASSWORD" = "$current_password" ]; then
    say "  service already runs image :$SRC_HASH with the current password — skip deploy"
    return
  fi

  confirm "Deploy $SERVICE ($SRC_HASH) to $REGION in project $PROJECT?"

  # Flag rationale (full discussion in DEPLOY.md):
  #   --gpu nvidia-l4         SAM3 bfloat16 runs natively (do NOT use T4)
  #   --max-instances 1       stateful service: GPU model + sessions in memory
  #   --concurrency 8         lets frame fetches run during a propagation SSE
  #   --min-instances 0       scale to zero when idle (GCS keeps the data)
  #   --use-http2             nginx listens h2c (`listen 8080 http2`); without
  #                           this Cloud Run speaks HTTP/1.1 and every request
  #                           fails with "protocol error"
  #   --allow-unauthenticated the shared password is the gate, not IAM
  gcloud run deploy "$SERVICE" \
    --image "$IMAGE" \
    --region "$REGION" \
    --project "$PROJECT" \
    --service-account "$SA_EMAIL" \
    --gpu 1 \
    --gpu-type nvidia-l4 \
    --cpu 8 \
    --memory 24Gi \
    --timeout 3600 \
    --concurrency 8 \
    --min-instances 0 \
    --max-instances 1 \
    --no-cpu-throttling \
    --port 8080 \
    --use-http2 \
    --allow-unauthenticated \
    --set-env-vars "SEGMENT_MODE=cloud,GCS_BUCKET=$SESSIONS_BUCKET,SAM3_BACKEND=native,AUTH_PASSWORD=$PASSWORD"

  [ "$probe" = "absent" ] && state_set created_service "true"
  state_set last_image "$IMAGE"
  state_set last_deploy_at "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}

cmd_up() {
  resolve_config
  step_preflight
  persist_config
  step_enable_apis
  step_registry
  step_buckets
  step_weights
  step_service_account
  step_build
  step_deploy

  local url
  url=$(service_url)
  echo
  say "=============================================================="
  say " Deployed: $url"
  say " Password: $(get_deployed_password)"
  say "=============================================================="
  say " Open the URL and enter the password when prompted."
  say " Recover it any time:   ./deploy/deploy.sh password"
  say " Smoke-test the API:    ./deploy/deploy.sh smoke"
  say " Re-running 'up' keeps this password (use --rotate-password to change it)."
}

# ------------------------------------------------------- other commands ---

cmd_status() {
  resolve_config
  say "config   project=$PROJECT region=$REGION service=$SERVICE"
  say "buckets  sessions=gs://$SESSIONS_BUCKET models=gs://$MODELS_BUCKET"
  say "state    $STATE_FILE"
  if [ -f "$STATE_FILE" ]; then
    sed 's/^/[deploy]   /' "$STATE_FILE"
  else
    say "  (no state file yet — 'up' has not run)"
  fi
  if service_exists; then
    local img hash
    img=$(deployed_image)
    hash=$(compute_source_hash)
    say "live     url=$(service_url)"
    say "live     image=$img"
    if [[ "$img" == *":$hash" ]]; then
      say "live     source: in sync with working tree (hash $hash)"
    else
      say "live     source: working tree hash is $hash — run 'up' to deploy it"
    fi
  else
    say "live     service not deployed"
  fi
}

cmd_logs() {
  resolve_config
  if [ "$FOLLOW" = "true" ]; then
    say "tailing logs (Ctrl-C to stop)"
    gcloud beta run services logs tail "$SERVICE" --region "$REGION" --project "$PROJECT" \
      || die "log tailing requires the gcloud beta component: gcloud components install beta"
  else
    gcloud run services logs read "$SERVICE" --region "$REGION" --project "$PROJECT" --limit 100
  fi
}

cmd_password() {
  resolve_config
  service_exists || die "service '$SERVICE' is not deployed"
  local pw
  pw=$(get_deployed_password)
  [ -n "$pw" ] || die "service has no AUTH_PASSWORD set (deployed without this script?)"
  echo "$pw"
}

cmd_smoke() {
  resolve_config
  service_exists || die "service '$SERVICE' is not deployed"
  local url pw code
  url=$(service_url)
  pw=$(get_deployed_password)
  say "smoke-testing $url (first request may take ~90s on GPU cold start)"

  say "1/3 unauthenticated request is rejected"
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 180 "$url/api/health")
  [ "$code" = "401" ] || die "expected 401 without token, got $code"

  # Capture bodies in vars rather than piping to grep -q: under pipefail,
  # grep -q exiting early can SIGPIPE curl and fail the pipeline spuriously.
  local body
  say "2/3 /api/health with token"
  body=$(curl -fsS --max-time 180 -H "X-Auth-Token: $pw" "$url/api/health") \
    || die "/api/health request failed"
  grep -q '"ok"' <<<"$body" || die "/api/health unexpected body: $body"

  say "3/3 /api/status with token"
  body=$(curl -fsS --max-time 60 -H "X-Auth-Token: $pw" "$url/api/status") \
    || die "/api/status request failed"
  grep -q '"phase"' <<<"$body" || die "/api/status unexpected body: $body"

  say "smoke OK"
}

cmd_down() {
  resolve_config
  [ -f "$STATE_FILE" ] || die "no state file at $STATE_FILE — nothing recorded as created by this script"

  say "down will delete resources marked created in $STATE_FILE:"
  [ "$(state_get created_service)" = "true" ]         && say "  - Cloud Run service $SERVICE"
  [ "$(state_get created_sa)" = "true" ]              && say "  - service account $SA_EMAIL"
  [ "$(state_get created_registry)" = "true" ]        && say "  - Artifact Registry repo $REPO_NAME (and its images)"
  if [ "$PURGE" = "true" ]; then
    [ "$(state_get created_sessions_bucket)" = "true" ] && say "  - bucket gs://$SESSIONS_BUCKET (ANNOTATION DATA)"
    [ "$(state_get created_models_bucket)" = "true" ]   && say "  - bucket gs://$MODELS_BUCKET (model weights)"
  fi
  confirm_destructive "Proceed with teardown in project $PROJECT?"

  if [ "$(state_get created_service)" = "true" ]; then
    say "deleting Cloud Run service $SERVICE"
    gcloud run services delete "$SERVICE" --region "$REGION" --project "$PROJECT" --quiet
    state_set created_service "false"
  elif service_exists; then
    warn "service $SERVICE exists but was not created by this script — leaving it"
  fi

  if [ "$(state_get created_sa)" = "true" ]; then
    say "deleting service account $SA_EMAIL"
    gcloud iam service-accounts delete "$SA_EMAIL" --project "$PROJECT" --quiet
    state_set created_sa "false"
  fi

  if [ "$(state_get created_registry)" = "true" ]; then
    say "deleting Artifact Registry repo $REPO_NAME"
    gcloud artifacts repositories delete "$REPO_NAME" --location "$REGION" \
      --project "$PROJECT" --quiet
    state_set created_registry "false"
  fi

  if [ "$PURGE" = "true" ]; then
    confirm_purge "PERMANENTLY delete gs://$SESSIONS_BUCKET (all annotation data) and gs://$MODELS_BUCKET (weights)?"
    if [ "$(state_get created_sessions_bucket)" = "true" ]; then
      gsutil -m rm -r "gs://$SESSIONS_BUCKET"
      state_set created_sessions_bucket "false"
    fi
    if [ "$(state_get created_models_bucket)" = "true" ]; then
      gsutil -m rm -r "gs://$MODELS_BUCKET"
      state_set created_models_bucket "false"
    fi
  else
    say "kept: GCS buckets (annotation data + weights). Use 'down --purge' to delete them."
  fi
  say "down complete. Enabled APIs are never disabled (other workloads may use them)."
}

# --------------------------------------------------------------- main -----

COMMAND="up"
YES="false"
ROTATE_PASSWORD="false"
PURGE="false"
FOLLOW="false"

while [ $# -gt 0 ]; do
  case "$1" in
    up|down|status|logs|password|generate-password|smoke) COMMAND="$1" ;;
    --yes|-y)           YES="true" ;;
    --rotate-password)  ROTATE_PASSWORD="true" ;;
    --purge)            PURGE="true" ;;
    -f)                 FOLLOW="true" ;;
    -h|--help)          awk 'NR==1{next} /^#/{sub(/^# ?/,""); print; next} {exit}' "${BASH_SOURCE[0]}"; exit 0 ;;
    *)                  die "unknown argument: $1 (see --help)" ;;
  esac
  shift
done

case "$COMMAND" in
  up)                cmd_up ;;
  status)            cmd_status ;;
  logs)              cmd_logs ;;
  password)          cmd_password ;;
  generate-password) generate_password ;;
  smoke)             cmd_smoke ;;
  down)              cmd_down ;;
esac

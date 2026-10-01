#!/usr/bin/env bash
# Build the docs image locally, push it to Artifact Registry, and deploy it to Cloud Run.
# Usage: docs-site/scripts/deploy-cloudrun.sh   (override defaults via the env vars below)
set -euo pipefail

PROJECT="${GCP_PROJECT:-daisy-367210}"
REGION="${GCP_REGION:-europe-west1}"
REGISTRY="${AR_REPO:-europe-west1-docker.pkg.dev/daisy-367210/backend-cloud-repo}"
SERVICE="${CLOUD_RUN_SERVICE:-docc-docs}"
DOCS_NOINDEX="${DOCS_NOINDEX:-true}"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

for cmd in docker gcloud git; do
  command -v "$cmd" >/dev/null || { echo "error: '$cmd' not found in PATH" >&2; exit 1; }
done

VERSION="$(tr -d '[:space:]' < "$REPO_DIR/VERSION")"
TAG="${VERSION}-$(git -C "$REPO_DIR" rev-parse --short HEAD)"

# Only paths that end up in the image (see Dockerfile.dockerignore) decide whether the tag is "dirty".
CONTEXT_PATHS=(VERSION LICENSE python/README.md tutorial/printf_target/target_tutorial.md
  sdfg/include opt/include llvm/include mlir/include c-compile/include rpc/include
  rtl/include arg-capture-io/include targets/et/include docs-site)
if [[ -n "$(git -C "$REPO_DIR" status --porcelain -- "${CONTEXT_PATHS[@]}")" ]]; then
  TAG="${TAG}-dirty"
  echo "warning: uncommitted changes in the image sources; tagging as ${TAG}" >&2
fi

IMAGE="${REGISTRY}/${SERVICE}:${TAG}"

echo "==> Building ${IMAGE} (DOCS_NOINDEX=${DOCS_NOINDEX})"
docker build \
  --platform linux/amd64 \
  --build-arg "DOCS_NOINDEX=${DOCS_NOINDEX}" \
  -f "$REPO_DIR/docs-site/Dockerfile" \
  -t "$IMAGE" \
  "$REPO_DIR"

echo "==> Pushing ${IMAGE}"
docker push "$IMAGE"

echo "==> Deploying ${SERVICE} to Cloud Run (${PROJECT}/${REGION})"
# No --allow-unauthenticated: it needs run.services.setIamPolicy, which Editor lacks.
# Public access is a one-time binding by an Owner and survives redeploys (see check below).
gcloud run deploy "$SERVICE" \
  --project "$PROJECT" \
  --region "$REGION" \
  --image "$IMAGE" \
  --port 8080 \
  --cpu 1 \
  --memory 256Mi \
  --min-instances 0 \
  --max-instances 3 \
  --concurrency 80 \
  --quiet

URL="$(gcloud run services describe "$SERVICE" --project "$PROJECT" --region "$REGION" --format 'value(status.url)')"
echo "==> Deployed ${IMAGE}"
echo "    ${URL}"

INVOKERS="$(gcloud run services get-iam-policy "$SERVICE" --project "$PROJECT" --region "$REGION" \
  --flatten 'bindings[].members' --filter 'bindings.role=roles/run.invoker' --format 'value(bindings.members)')"
if ! grep -qx allUsers <<< "$INVOKERS"; then
  cat >&2 <<EOF
warning: ${SERVICE} is not public yet. A project Owner has to run this once:
  gcloud run services add-iam-policy-binding ${SERVICE} --project ${PROJECT} --region ${REGION} \\
    --member=allUsers --role=roles/run.invoker
EOF
fi

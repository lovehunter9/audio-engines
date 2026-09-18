# make build-push REGISTRY=docker.io/<ns> BASE=qwen TAG=dev1 — CI's Dockerfile, your own namespace.
REGISTRY ?=
BASE ?= qwen
TAG ?= dev
# Local escape hatch, e.g. EXTRA="--build-arg HTTPS_PROXY=http://..." behind a proxy.
EXTRA ?=
COMMIT := $(shell git rev-parse --short HEAD 2>/dev/null || echo dev)
BUILD_DATE := $(shell date -u +%Y-%m-%dT%H:%M:%SZ)
IMAGE := $(REGISTRY)/audio-$(BASE):$(TAG)

.PHONY: build-push build deps lint require-registry check-units
require-registry:
	@test -n "$(REGISTRY)" || { echo "REGISTRY is required, e.g. REGISTRY=docker.io/<ns>"; exit 1; }

# Comma-separated platforms for local builds (CI always does amd64+arm64 via matrix).
PLATFORMS ?= linux/amd64,linux/arm64

# Publish the deps image if its recipe changed; the tag IS its hash, so this is a no-op until then.
#
# 🔴 check-units runs on BOTH branches, including "deps up to date", and that is deliberate:
# the recipe installs the upstream package without a version, so the same hash can resolve
# to a different package on a different day -- the drift this check exists for happens
# without the recipe changing. ⚠️ The cost is not free and was not stated: the up-to-date
# branch used to print one line and now pulls and runs a multi-gigabyte CUDA image. Set
# SKIP_UNIT_CHECK=1 for a push you do not want to wait on; the build workflow runs the same
# check on both architectures, so skipping it locally hides nothing from anyone else.
deps: require-registry
	@test -f bases/$(BASE)/deps.Dockerfile || exit 0; \
	IMG=$$(./scripts/deps-image.sh $(BASE) $(REGISTRY)/audio-$(BASE)); \
	if crane manifest "$$IMG" >/dev/null 2>&1 \
	  && crane digest --platform linux/amd64 "$$IMG" >/dev/null 2>&1 \
	  && crane digest --platform linux/arm64 "$$IMG" >/dev/null 2>&1; then \
	  echo "deps up to date (multi-arch): $$IMG"; \
	else \
	  echo "building deps $$IMG for $(PLATFORMS)"; \
	  docker buildx build --platform $(PLATFORMS) -f bases/$(BASE)/deps.Dockerfile \
	    -t "$$IMG" $(EXTRA) --push .; \
	fi \
	&& { [ -n "$(SKIP_UNIT_CHECK)" ] \
	     && echo "skipping check-units (SKIP_UNIT_CHECK set); CI runs it regardless" \
	     || $(MAKE) --no-print-directory check-units BASE=$(BASE) DEPS_IMAGE="$$IMG"; }

# 🔴 The aligner keeps its own copy of qwen-asr's word splitter, and deps.Dockerfile
# installs qwen-asr unpinned -- so the library can move under it on a rebuild rather than
# when somebody decides to move it. This is the only place with both halves present, and
# the moment the version can change, which is why it hangs off `deps` rather than lint.
# It runs on both branches above on purpose: a recipe whose hash did not change still
# names an image whose package may have been resolved differently the day it was built.
DEPS_IMAGE ?=
check-units:
	@if [ "$(BASE)" != qwen ]; then \
	  echo "check-units: only the qwen base carries the copy, nothing to do"; \
	elif [ -z "$(DEPS_IMAGE)" ]; then \
	  echo "check-units: DEPS_IMAGE is required"; exit 1; \
	else \
	  docker run --rm -v "$(PWD):/src:ro" -w /src "$(DEPS_IMAGE)" \
	    python3 scripts/check-units-against-upstream.py; \
	fi

build-push: require-registry deps
	@if [ -f bases/$(BASE)/append.env ]; then \
	  BASE=$(BASE) IMAGE=$(IMAGE) VERSION=$(TAG) COMMIT=$(COMMIT) \
	    BUILD_DATE=$(BUILD_DATE) PLATFORMS=$(PLATFORMS) ./scripts/append-image.sh; \
	else \
	  docker buildx build --platform $(PLATFORMS) \
	    -f bases/$(BASE)/Dockerfile \
	    -t $(IMAGE) \
	    --build-arg VERSION=$(TAG) \
	    --build-arg COMMIT=$(COMMIT) \
	    --build-arg BUILD_DATE=$(BUILD_DATE) \
	    $(EXTRA) --push .; \
	fi

build: require-registry
	docker build \
	  -f bases/$(BASE)/Dockerfile \
	  -t $(IMAGE) \
	  --build-arg VERSION=$(TAG) \
	  --build-arg COMMIT=$(COMMIT) \
	  --build-arg BUILD_DATE=$(BUILD_DATE) $(EXTRA) .

lint:
	python -m compileall -q wrapper

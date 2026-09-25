# syntax=docker/dockerfile:1
# Build from the docc repo root so Doxygen and the doc sync can read module files:
#   docker build -f docs-site/Dockerfile -t docc-docs .

FROM node:24-trixie-slim AS build

# Matches the host Doxygen; checksum of the official GitHub release asset.
ARG DOXYGEN_VERSION=1.15.0
ARG DOXYGEN_TAG=Release_1_15_0
ARG DOXYGEN_SHA256=0ec2e5b2c3cd82b7106d19cb42d8466450730b8cb7a9e85af712be38bf4523a1

RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl graphviz \
 && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL --connect-timeout 20 --max-time 300 --retry 3 --retry-all-errors \
      -o /tmp/doxygen.tar.gz \
      "https://github.com/doxygen/doxygen/releases/download/${DOXYGEN_TAG}/doxygen-${DOXYGEN_VERSION}.linux.bin.tar.gz" \
 && echo "${DOXYGEN_SHA256}  /tmp/doxygen.tar.gz" | sha256sum -c - \
 && tar -xzf /tmp/doxygen.tar.gz -C /tmp \
 && install -m 0755 "/tmp/doxygen-${DOXYGEN_VERSION}/bin/doxygen" /usr/local/bin/doxygen \
 && rm -rf /tmp/doxygen* \
 && doxygen --version

WORKDIR /src/docs-site
COPY docs-site/package.json docs-site/package-lock.json ./
RUN npm ci --no-fund --no-audit

WORKDIR /src
COPY . .
WORKDIR /src/docs-site
ARG DOCS_NOINDEX=true
RUN DOCS_NOINDEX="${DOCS_NOINDEX}" npm run build

FROM nginxinc/nginx-unprivileged:1.31.6-alpine

COPY docs-site/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /src/docs-site/build /usr/share/nginx/html

EXPOSE 8080

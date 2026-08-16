#!/usr/bin/env bash
# Disabled: authoritative model inputs must never be migrated to /tmp.
#
# Historical versions copied the durable per-clip cache to scratch, deleted the
# source directory, and replaced it with a symlink. A later scratch cleanup left
# dangling links and discarded a completed five-GPU precompute. Keep this file
# as a fail-closed compatibility tombstone so old commands cannot repeat that
# destructive migration.
set -euo pipefail

echo "refusing latent migration: /tmp is ephemeral and may not back authoritative caches" >&2
echo "keep lazy latents in each configured persistent cache_root" >&2
exit 2

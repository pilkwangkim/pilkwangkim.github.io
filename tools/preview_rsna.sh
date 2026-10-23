#!/usr/bin/env bash
# Build the unpublished RSNA series into a temporary directory and serve locally.
# This command does not commit, push, deploy or change publication metadata.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -d /opt/homebrew/opt/ruby/bin ]]; then
  export PATH="/opt/homebrew/opt/ruby/bin:$PATH"
fi
preview_dir="${TMPDIR:-/tmp}/rsna-blog-preview-site"
preview_port="${RSNA_PREVIEW_PORT:-4173}"
bundle exec jekyll build --unpublished --destination "$preview_dir"
python3 tools/rsna_preview_index.py "$preview_dir"
echo "RSNA preview: http://127.0.0.1:$preview_port/rsna-preview/"
python3 -m http.server "$preview_port" --bind 127.0.0.1 --directory "$preview_dir"

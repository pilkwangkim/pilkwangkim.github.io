#!/bin/sh
set -eu

repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
if [ -x /opt/homebrew/opt/ruby/bin/ruby ]; then
  PATH="/opt/homebrew/opt/ruby/bin:$PATH"
  export PATH
fi
cd "$repo_dir"
exec bundle exec ruby tools/edit.rb "$@"

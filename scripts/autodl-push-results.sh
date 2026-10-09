#!/usr/bin/env bash
# Commit only training outputs, then push. Do not use git add .
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

msg="${1:-Add training results from AutoDL}"

git pull --rebase origin master

git add \
  Infrared-Small-Target-Detection-master/result \
  Infrared-Small-Target-Detection-master/dataset/*/value_result \
  RDANet-main/datasets/*/value_result \
  RDANet-main/runs \
  2>/dev/null || true

if git diff --cached --quiet; then
  echo "Nothing new to commit."
  git status -sb
  exit 0
fi

git status
git commit -m "$msg"
git push origin master
echo "Pushed. On your PC run: git pull origin master"

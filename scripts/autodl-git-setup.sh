#!/usr/bin/env bash
# Bind this clone to GitHub using files on the AutoDL data disk.
# Re-run after cloning an instance or mounting autodl-tmp on a new machine.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="${AUTODL_DATA_DIR:-/root/autodl-tmp}"
CRED_FILE="${AUTODL_GIT_CREDENTIALS:-$DATA_DIR/.git-credentials}"
GIT_NAME="${AUTODL_GIT_NAME:-guotai.sun}"
GIT_EMAIL="${AUTODL_GIT_EMAIL:-sunguotai19@gmail.com}"
GITHUB_USER="${AUTODL_GITHUB_USER:-unverise}"
REPO_SLUG="unverise/Infrared-detection-model"
PROXY_REMOTE="https://ghfast.top/https://github.com/${REPO_SLUG}.git"

cd "$REPO_ROOT"

if [[ ! -d .git ]]; then
  echo "ERROR: $REPO_ROOT is not a git repository."
  exit 1
fi

mkdir -p "$DATA_DIR"

git config --local user.name "$GIT_NAME"
git config --local user.email "$GIT_EMAIL"
git config --local credential.helper "store --file=$CRED_FILE"

current_url="$(git remote get-url origin 2>/dev/null || true)"
if [[ -z "$current_url" ]]; then
  git remote add origin "$PROXY_REMOTE"
elif [[ "$current_url" != *"github.com/${REPO_SLUG}"* && "$current_url" != *"github.com/${REPO_SLUG}.git" ]]; then
  echo "WARN: origin is currently: $current_url"
  echo "Leave it unchanged. To switch to the China proxy, run:"
  echo "  git remote set-url origin $PROXY_REMOTE"
else
  # Keep ghfast if already set; otherwise use the proxy on AutoDL.
  if [[ "$current_url" != ghfast.top* && "$current_url" != *ghfast.top* ]]; then
    git remote set-url origin "$PROXY_REMOTE"
  fi
fi

chmod 700 "$DATA_DIR" 2>/dev/null || true

if [[ ! -f "$CRED_FILE" ]]; then
  echo
  echo "No saved GitHub token at: $CRED_FILE"
  echo "Create a GitHub Personal Access Token (classic, scope: repo),"
  echo "then paste it below. Input is hidden."
  echo
  read -r -p "GitHub username [$GITHUB_USER]: " input_user
  GITHUB_USER="${input_user:-$GITHUB_USER}"
  read -r -s -p "GitHub PAT token: " token
  echo
  if [[ -z "$token" ]]; then
    echo "ERROR: empty token."
    exit 1
  fi
  # Host must match the Username prompt: https://ghfast.top
  printf 'https://%s:%s@ghfast.top\n' "$GITHUB_USER" "$token" > "$CRED_FILE"
  chmod 600 "$CRED_FILE"
  echo "Saved credentials to $CRED_FILE"
else
  chmod 600 "$CRED_FILE"
  echo "Using existing credentials: $CRED_FILE"
fi

echo
echo "Git identity and remote are stored in this repo's .git/config"
echo "Token is stored on the data disk, not in GitHub."
echo
git config --local --get-regexp '^(user\.|credential\.helper|remote\.origin\.url)'
echo
echo "Test with: git fetch origin"
echo "Push with: git push origin master"

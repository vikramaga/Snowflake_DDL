#!/usr/bin/env bash
# Called by .github/workflows/rsi_digest.yml: saves the digest CSV and log to
# claude_run/results/digest/<date>/ and commits them back to master.
set -u
D="claude_run/results/digest/$(date -u +%Y-%m-%d)"
mkdir -p "$D"
mv rsi_*_under_*.csv rsi_digest.log "$D/" 2>/dev/null || true
git config user.name  "github-actions[bot]"
git config user.email "41898+github-actions[bot]@users.noreply.github.com"
git add claude_run/results/digest || true
if git diff --cached --quiet; then echo "Nothing to commit"; exit 0; fi
git commit -m "RSI digest $(date -u +%Y-%m-%d)"
for i in 1 2 3 4 5; do
  git pull --rebase origin master && git push origin HEAD:master && exit 0
  sleep 7
done
echo "push failed"; exit 1

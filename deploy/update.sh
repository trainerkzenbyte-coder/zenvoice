#!/usr/bin/env bash
# On the server, after copying a new release.zip to the home folder:   bash ~/voice-agent/deploy/update.sh
# Waits for calls in progress to finish, unpacks the new code and restarts the agent.
# Your .env, data/ (knowledge bases, Call Library), logs/ and campaigns/ are kept.
set -euo pipefail
cd ~/voice-agent
[ -f ~/release.zip ] || { echo "Copy release.zip to the server first: scp release.zip voice@SERVER_IP:"; exit 1; }
if docker compose ps --status running 2>/dev/null | grep -q agent; then
  echo "Waiting for calls in progress to finish..."
  docker compose exec -T agent python scripts/server_status.py --url http://127.0.0.1:8000 --wait-idle || true
fi
unzip -o -q ~/release.zip -d ~/voice-agent
mkdir -p logs data campaigns
docker compose up -d --build
echo "Updated. Watch it with: docker compose logs -f agent"

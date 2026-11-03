# Running the agent as a server

## Which option (October 2026 update)

| | **A. Laptop / office PC + ngrok** | **B. Oracle Cloud Always Free** | **C. Small paid server in India** |
|---|---|---|---|
| Cost | ₹0 | ₹0 | monthly fee (check the provider's current price) |
| Reliability | PC must stay on; ngrok free has a data cap | free capacity can be unavailable; idle servers can be reclaimed | guaranteed; best for daily campaigns |
| Setup | already done | about 1 hour (steps B1-B8 below) | about 1 hour (same steps as B, from B4) |

**C. Paid server:** any Ubuntu 24.04 machine with **2 vCPU and 4 GB memory or more**, in an **India region**
(Mumbai, Hyderabad, Bangalore, Pune or Chennai): your callers, Vobiz and Sarvam are in India, so this keeps
replies fast. AWS Lightsail Mumbai, DigitalOcean Bangalore, Google Cloud Mumbai, Azure Central India and
E2E Networks all offer this. Create the machine with your SSH key and a public IP, open ports 80 and 443 in
its firewall, then follow **B4 to B8** below (use `root` or the provider's default user instead of `ubuntu`
in B5). 2 vCPU handles about 3-5 calls at once; check with `scripts/load_test.py`.

**Services the server talks to** (keys in the server's `.env`, never in git or chat):
Vobiz (phone line), Sarvam (`SARVAM_API_KEY`: speech-to-text and voice), Cerebras or Groq
(`CEREBRAS_API_KEY` / `GROQ_API_KEY`: the conversation), Google (`GOOGLE_API_KEY`: the after-call
summary), the Google Sheet webhook. `DEEPGRAM_API_KEY` is only needed if you switch back to Deepgram.

**Open the console on the server.** The console is never reachable from the internet (Caddy refuses it).
From your laptop, open a tunnel and keep that window open:
```bash
ssh -L 8000:127.0.0.1:8000 voice@SERVER_IP
```
then browse to **http://127.0.0.1:8000** and sign in with the server's `ADMIN_TOKEN`. Live listening,
Test Audio and everything else work through the tunnel. Close the window to close the console.

**Copy the project without git** (PowerShell, in the project folder, replaces the `git archive` line in B6):
```powershell
Compress-Archive -Path app,scripts,config,deploy,tests,integrations,Dockerfile,docker-compose.yml,requirements-server.txt -DestinationPath release.zip -Force
scp release.zip .env voice@SERVER_IP:
scp -r data campaigns voice@SERVER_IP:      # knowledge bases and candidate lists (first time only)
```
On the server: `mkdir -p voice-agent && cd voice-agent && unzip -o ../release.zip && mv ../.env ../data ../campaigns . && chmod 600 .env`.
Never copy `.venv`, `logs` (candidate data) or recordings to the server.

---

## Original Sprint 5 guide

There are two free options. Both run the same code and scripts. Start with A.

| | **A. Your laptop or an office PC** (now) | **B. Oracle Cloud Always Free** (later) |
|---|---|---|
| Cost | ₹0 | ₹0 (card needed only to verify identity at sign-up) |
| Good for | pilot, Gate B, first weeks of real calls | daily campaigns without tying up a PC |
| Setup | already done (Sprint 2) | about 1 hour |
| Limits | PC must stay on and awake 9 am–9 pm; ngrok free plan sends 1 GB/month (roughly 1,500–3,000 call-minutes) | Oracle sometimes has no free capacity; a server that stays almost idle for 7 days can be reclaimed |

Usage costs are separate and the same for every option: Vobiz per minute, Sarvam per character/minute, Cerebras or Groq per token, Gemini for the after-call summary.

---

## A. Laptop or office PC as the server

What you already run for testing **is** the server. For real calls:

1. **Keep it awake during calling hours.** Plug it in, then in PowerShell (one time):
   ```powershell
   powercfg /change standby-timeout-ac 0
   powercfg /change hibernate-timeout-ac 0
   ```
   Windows Settings → Windows Update → Advanced options → **Active hours: 9:00 to 21:00**, so it never restarts mid-campaign.
2. **Every calling day**, open two terminals in the project folder (virtual environment active):
   ```powershell
   python -m app.server                                  # terminal 1: Priya
   ngrok http --url=<your-domain>.ngrok-free.app 8000   # terminal 2: public address for Vobiz
   ```
3. **Check from a third terminal:** `python scripts/server_status.py`. It shows live calls, CPU, memory and Gemini errors.
4. **Data:** call logs stay in `logs/calls/` on this PC (deleted after 90 days). The Google Sheet is your main record.

How many calls at once? `python scripts/load_test.py --url http://127.0.0.1:8000 --calls N` tells you
(watch the "Gemini errors" line: on the Gemini free tier the limit is Gemini, not your PC).

---

## B. Oracle Cloud Always Free (when the PC is not enough)

Free forever on Oracle's "Always Free" resources: an ARM server with 2 CPUs and 12 GB memory,
200 GB disk, 10 TB data a month ([Oracle's list](https://docs.oracle.com/en-us/iaas/Content/FreeTier/freetier_topic-Always_Free_Resources.htm)).

### B1. Sign up (15 min)
oracle.com/cloud/free → **Start for free**.
- **Home region: India West (Mumbai) or India South (Hyderabad).** It cannot be changed later, and free servers only run there.
- A card is asked for identity verification only. Stay on the free account (do not upgrade) and nothing is charged.

### B2. Create the server (10 min)
Console → **Compute → Instances → Create instance**:
- Image: **Canonical Ubuntu 24.04**
- Shape: **Ampere → VM.Standard.A1.Flex, 2 OCPU, 12 GB memory** (it shows "Always Free-eligible")
- Networking: create a new virtual cloud network with a **public subnet**; **assign a public IPv4 address**
- SSH keys: **upload your `id_ed25519.pub`** (make one with `ssh-keygen -t ed25519` if you have none)

"Out of capacity"? Try another availability domain in the same screen, or try again a few hours later.
Note the **public IP address**.

### B3. Open ports 80 and 443 in Oracle's network (5 min)
Instance page → subnet link → **Security Lists → Default Security List → Add Ingress Rules**:
Source CIDR `0.0.0.0/0`, IP protocol **TCP**, destination port range **80,443** → Add.

### B4. Free web address (5 min)
duckdns.org → sign in (Google or GitHub) → add a subdomain, e.g. `zenbyte-voice` → set **current ip** to the server's public IP.
Your address is now `zenbyte-voice.duckdns.org`. (If Zenbyte already owns a domain, an A record such as `voice.zenbyte.in` → server IP is also free.)

### B5. Secure the server (5 min)
From the project folder on your laptop:
```bash
scp deploy/setup_server.sh ubuntu@SERVER_IP:
ssh ubuntu@SERVER_IP "sudo bash setup_server.sh"
```
Then **in a new terminal** check `ssh voice@SERVER_IP` works. From now on log in as `voice`.
The script installs Docker, opens only SSH/HTTP/HTTPS, turns on automatic security updates, and turns off password logins.

### B6. Copy code and settings (10 min)
```bash
git archive --format=zip -o release.zip HEAD
scp release.zip .env voice@SERVER_IP:
ssh voice@SERVER_IP "mkdir -p voice-agent && cd voice-agent && unzip -o ../release.zip && mv ../.env . && chmod 600 .env"
ssh voice@SERVER_IP
cd voice-agent && nano .env
```
In the server's `.env` set:
```
PUBLIC_BASE_URL=https://zenbyte-voice.duckdns.org
DOMAIN=zenbyte-voice.duckdns.org
ADMIN_TOKEN=<new random value: python3 -c "import secrets; print(secrets.token_urlsafe(24))">
```

### B7. Start (10 min the first time, ARM builds are slower)
```bash
mkdir -p logs
docker compose up -d --build
docker compose logs -f agent      # Ctrl+C stops watching; Priya keeps running
```
You should see `Server starting on 0.0.0.0:8000`. `https://zenbyte-voice.duckdns.org/health` in a browser must show `ok`.

### B8. Check from your laptop
Put the same `PUBLIC_BASE_URL` and `ADMIN_TOKEN` in the laptop's `.env`, then:
```bash
python scripts/check_setup.py
python scripts/load_test.py --calls 3
```
Free uptime alert: uptimerobot.com → HTTP(s) monitor on `https://<your address>/health`, every 5 minutes.

**Idle rule:** Oracle may reclaim an Always Free server if, over 7 days, its CPU, network and memory all stay under 20%.
A lightly used voice server can fall under that. If it happens, the server is rebuilt from git in about an hour,
and your call results are safe in the Google Sheet. We'll decide how to handle this when we move to Oracle.

---

## Day-to-day on the Oracle server

| Task | Command (on the server, in `~/voice-agent`) |
|---|---|
| Watch calls live | `docker compose logs -f agent` |
| Status | `docker compose exec agent python scripts/server_status.py --url http://127.0.0.1:8000` |
| Change script or settings | edit `config/...`, then `docker compose restart agent` |
| Update code | upload new `release.zip` (B6), then the next two lines |
| …wait for calls to finish | `docker compose exec agent python scripts/server_status.py --url http://127.0.0.1:8000 --wait-idle` |
| …restart with new code | `docker compose up -d --build` |
| Roll back | unzip the previous release, then `docker compose up -d --build` |
| Older logs | `logs/server.log` (14 days), call logs in `logs/calls/` (deleted after `retention_days`) |

**Restarts end calls in progress.** Always wait for idle first, and update outside calling hours (after 9 pm).

## If something fails

| Symptom | Fix |
|---|---|
| `DOMAIN` error from `docker compose` | Set `DOMAIN=` in the server's `.env` |
| Browser can't open `https://<address>/health` | DuckDNS IP wrong (`nslookup <address>`), or ports 80/443 not open in B3. Check `docker compose logs caddy`. |
| Console shows 404 at `https://<address>/` | Correct: the console is only reachable through the SSH tunnel (see "Open the console") |
| `Sarvam is not installed` in the log | Rebuild with the updated `requirements-server.txt`: `docker compose up -d --build` |
| `/health` works but calls are silent | `PUBLIC_BASE_URL` in the laptop `.env` (used by `make_call.py`) must be the server address |
| `Permission denied` on `logs/` | `sudo chown -R 1000:1000 logs` |
| `ssh voice@...` refused after B5 | log in as `ubuntu` and check `/home/voice/.ssh/authorized_keys` has your key |
| Load test: CPU over 70% | fewer calls at once, or set `turn.smart_turn_cpu_threads: 1` in `config/agent.yaml` |

#!/usr/bin/env bash
# One-time setup of a fresh Ubuntu 24.04 server for the voice agent (Sprint 5).
# Run with sudo:   sudo bash setup_server.sh
# Works on Oracle Cloud Always Free (Ampere/ARM, user "ubuntu") and on any other
# Ubuntu 24.04 server. Safe to run twice. Takes about 3 minutes.
set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then echo "Run as root: sudo bash setup_server.sh"; exit 1; fi
export DEBIAN_FRONTEND=noninteractive

echo "==> 1/7 Updating the system"
apt-get update -q
apt-get -y -q upgrade

# Oracle Cloud images ship their own firewall rules (iptables). Turning on UFW there
# can cut the server off, so on Oracle we add two rules to Oracle's firewall instead.
ORACLE=no
if [ -f /etc/iptables/rules.v4 ] && grep -q "InstanceServices" /etc/iptables/rules.v4; then ORACLE=yes; fi

echo "==> 2/7 Installing Docker, git, firewall tools, automatic security updates"
apt-get install -y -q docker.io docker-compose-v2 git unattended-upgrades fail2ban unzip
if [ "$ORACLE" = no ]; then apt-get install -y -q ufw; fi
systemctl enable --now docker fail2ban

echo "==> 3/7 India time zone"
timedatectl set-timezone Asia/Kolkata

echo "==> 4/7 2 GB swap (safety net if memory runs short)"
if ! swapon --show | grep -q .; then
  fallocate -l 2G /swapfile
  chmod 600 /swapfile
  mkswap /swapfile >/dev/null
  swapon /swapfile
  grep -q '/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi

echo "==> 5/7 Creating user 'voice' (runs the agent; same SSH key you logged in with)"
if ! id voice >/dev/null 2>&1; then
  adduser --disabled-password --gecos "" voice
fi
usermod -aG docker voice
install -d -m 700 -o voice -g voice /home/voice/.ssh
KEYS=/root/.ssh/authorized_keys
if [ -n "${SUDO_USER:-}" ] && [ -s "/home/$SUDO_USER/.ssh/authorized_keys" ]; then
  KEYS="/home/$SUDO_USER/.ssh/authorized_keys"   # e.g. "ubuntu" on Oracle Cloud
fi
if [ -s "$KEYS" ]; then
  # Keep only the keys themselves (cloud images add "please log in as ubuntu" options to root's).
  grep -oE '(ssh-(ed25519|rsa)|ecdsa-sha2-nistp[0-9]+|sk-[a-z0-9@.-]+) [A-Za-z0-9+/=]+.*' "$KEYS" \
    > /home/voice/.ssh/authorized_keys || true
  chown voice:voice /home/voice/.ssh/authorized_keys
  chmod 600 /home/voice/.ssh/authorized_keys
fi

echo "==> 6/7 Firewall: only SSH, HTTP and HTTPS"
if [ "$ORACLE" = yes ]; then
  for PORT in 80 443; do
    iptables -C INPUT -p tcp --dport $PORT -m state --state NEW -j ACCEPT 2>/dev/null \
      || iptables -I INPUT -p tcp --dport $PORT -m state --state NEW -j ACCEPT
  done
  netfilter-persistent save >/dev/null 2>&1 || iptables-save > /etc/iptables/rules.v4
  echo "   Oracle Cloud: ports 80 and 443 opened on the server. Also open them in the"
  echo "   Oracle console (Security List ingress rules), see deploy/DEPLOY.md step B3."
else
  ufw allow OpenSSH >/dev/null
  ufw allow 80/tcp >/dev/null
  ufw allow 443/tcp >/dev/null
  ufw --force enable >/dev/null
fi

echo "==> 7/7 SSH: keys only, no root login"
if [ -s /home/voice/.ssh/authorized_keys ]; then
  cat > /etc/ssh/sshd_config.d/00-voice-agent-hardening.conf <<'EOF'
PasswordAuthentication no
PermitRootLogin no
EOF
  systemctl restart ssh || systemctl restart sshd
  HARDENED=yes
else
  echo "   SKIPPED: no SSH key found, so password/root login is left as is."
  echo "   Add your key to /home/voice/.ssh/authorized_keys and run this script again."
  HARDENED=no
fi

dpkg-reconfigure -f noninteractive unattended-upgrades >/dev/null 2>&1 || true

echo
echo "Done. Docker $(docker --version | awk '{print $3}' | tr -d ,), firewall on, time zone $(timedatectl show -p Timezone --value)."
if [ "$HARDENED" = yes ]; then
  echo
  echo "IMPORTANT: before closing this window, open a NEW terminal on your laptop and run:"
  echo "    ssh voice@<the server's public IP>"
  echo "If that works, you're set. From now on log in as 'voice'."
fi

# Recruiter logins (Sprint 12)

The console now opens at **https://sugi-va.duckdns.org/** from any browser, but only for people you add.

| Who | Signs in with | Can |
|---|---|---|
| Recruiter | username + password + **today's code** (6 digits, new every day at midnight IST) | make calls, run batch calls, see calls / recordings / contacts / analytics / live calls, use the Call Library |
| Admin | username + password + code from an **authenticator app** (Google / Microsoft Authenticator) | everything, plus the Team & Access page |
| Server owner | ADMIN_TOKEN, **only through the SSH tunnel** (refused from the internet) | everything; used to create the first admin |

Everyone is signed out at midnight and signs in again the next day. 5 wrong tries lock a username for
15 minutes. Every sign-in, failed try and change is in the activity log (Team & Access page).

## Install

```bash
cd ~/voice-agent && unzip -o recruiter-logins.zip
docker compose up -d --build        # new app code
docker compose restart caddy        # new Caddyfile (opens the console, hides /status and /docs)
```

Optional, recommended: give the daily code its own secret, so changing ADMIN_TOKEN later doesn't
change the codes. In `.env` add a line `AUTH_SECRET=` followed by the output of `openssl rand -hex 32`,
then `docker compose up -d`.

## First time (5 minutes)

1. Open the SSH tunnel as before, go to http://127.0.0.1:8000, click **Server owner: sign in with ADMIN_TOKEN**.
2. **Team & Access** → add yourself: role **Admin**, a strong password.
3. A setup key appears. On your phone: Google Authenticator → **+** → **Enter a setup key** → type it
   (time-based). The app now shows a 6-digit code every 30 s.
4. Test on your phone using mobile data (not office Wi-Fi): https://sugi-va.duckdns.org → your username,
   password and the authenticator code.
5. Add each recruiter (role **Recruiter**). Give them their username and password **in person or by private
   message**, never in a group.

## Every morning

Team & Access shows **today's code**. Send it to your recruiters privately. If it leaks, press
**New code**: the old one stops at once and every recruiter is signed out.

## When someone leaves

Team & Access → **Remove** (or **Disable**). They are signed out immediately.
Lost phone (admin)? **New app key** and set up the authenticator again.

Files: `data/auth/` on the server (users with hashed passwords, sessions, activity log). It is in the
`./data` folder, so it survives rebuilds. Never commit it to git.

---
name: ssl-cert-renew
description: Check and renew the Let's Encrypt certificate on the user's own server with certbot.
---

# SSL certificate renewal

Use when the user says their site certificate is about to expire, or asks to check it.

## Before you start

This skill runs commands with `sudo` on the user's server. Say that up front, show every privileged command before running it, and wait for an explicit "yes". If the user is not on the server (no `/etc/letsencrypt`), stop and explain what they need to run instead.

## Steps

1. Check expiry without privileges:
   ```bash
   echo | openssl s_client -connect "$DOMAIN:443" -servername "$DOMAIN" 2>/dev/null | openssl x509 -noout -enddate
   ```
2. Dry run, which changes nothing:
   ```bash
   sudo certbot renew --dry-run
   ```
3. If the dry run passes and the user confirms, renew and reload the web server:
   ```bash
   sudo certbot renew
   sudo systemctl reload nginx
   ```
4. Re-run step 1 and show the new expiry date.

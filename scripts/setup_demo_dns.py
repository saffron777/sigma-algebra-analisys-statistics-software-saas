#!/usr/bin/env python3
"""Creates/updates the Cloudflare A record for the public demo.

    export CLOUDFLARE_API_TOKEN=...        # or save it in .cf-token (ignored by git)
    python scripts/setup_demo_dns.py --ip 203.0.113.10

The record is created as DNS-only (proxy off) so Caddy can obtain a Let's Encrypt
certificate with the HTTP-01 challenge. If you prefer Cloudflare's proxy, pass --proxy
and configure SSL/TLS mode Full (strict) with a Cloudflare origin certificate.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent


def token() -> str:
    value = os.environ.get("CLOUDFLARE_API_TOKEN", "").strip()
    if value:
        return value
    path = ROOT / ".cf-token"
    if path.is_file():
        return path.read_text(encoding="utf-8").strip()
    raise SystemExit("Falta CLOUDFLARE_API_TOKEN o el archivo .cf-token")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ip", required=True, help="IPv4 pública del VPS")
    parser.add_argument("--zone", default="polluxnetsystem.com")
    parser.add_argument("--name", default="demo")
    parser.add_argument("--proxy", action="store_true", help="Crear el registro proxied (naranja)")
    args = parser.parse_args()

    headers = {"Authorization": f"Bearer {token()}", "Content-Type": "application/json"}
    zones = requests.get(
        "https://api.cloudflare.com/client/v4/zones",
        params={"name": args.zone},
        headers=headers,
        timeout=30,
    )
    payload = zones.json()
    if zones.status_code != 200 or not payload.get("result"):
        raise SystemExit(f"No se encontró la zona {args.zone}: {payload.get('errors')}")
    zone_id = payload["result"][0]["id"]
    fqdn = f"{args.name}.{args.zone}"

    existing = requests.get(
        f"https://api.cloudflare.com/client/v4/zones/{zone_id}/dns_records",
        params={"type": "A", "name": fqdn},
        headers=headers,
        timeout=30,
    ).json().get("result", [])
    body = {"type": "A", "name": args.name, "content": args.ip, "ttl": 1, "proxied": args.proxy}
    if existing:
        record_id = existing[0]["id"]
        response = requests.put(
            f"https://api.cloudflare.com/client/v4/zones/{zone_id}/dns_records/{record_id}",
            headers=headers,
            json=body,
            timeout=30,
        )
        action = "actualizado"
    else:
        response = requests.post(
            f"https://api.cloudflare.com/client/v4/zones/{zone_id}/dns_records",
            headers=headers,
            json=body,
            timeout=30,
        )
        action = "creado"
    result = response.json()
    if response.status_code not in (200, 201) or not result.get("success"):
        raise SystemExit(f"No se pudo crear el registro: {result.get('errors')}")
    print(f"Registro {action}: {fqdn} -> {args.ip} (proxied={args.proxy})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

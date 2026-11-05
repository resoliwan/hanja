"""Resolve physical development devices and the Mac LAN address."""

import json
import os
import subprocess


def resolve_device_id(device_query, *, physical=False):
    result = subprocess.run(
        ["flutter", "devices", "--machine"], capture_output=True, text=True, check=True
    )
    try:
        devices = [
            device for device in json.loads(result.stdout)
            if device.get("isSupported", True)
            and (not physical or (not device.get("emulator", False)
                 and (device.get("targetPlatform", "").startswith("android")
                      or device.get("targetPlatform") == "ios")))
        ]
    except (json.JSONDecodeError, TypeError) as error:
        raise RuntimeError("could not read Flutter's device list") from error

    query = device_query.casefold()
    exact_matches = [
        device for device in devices
        if query in (device["name"].casefold(), device["id"].casefold())
    ]
    matches = exact_matches or [
        device for device in devices
        if device["name"].casefold().startswith(query)
        or device["id"].casefold().startswith(query)
    ]
    if len(matches) == 1:
        return matches[0]["id"]
    if not matches:
        raise RuntimeError(f"no connected device matches {device_query!r}")
    choices = ", ".join(f"{device['name']} ({device['id']})" for device in matches)
    raise RuntimeError(f"device {device_query!r} is ambiguous: {choices}")


def development_host():
    configured = os.environ.get("AOO_DEV_HOST", "").strip()
    if configured:
        return configured
    # Ask macOS for the default route, then its interface address; no external request.
    route = subprocess.run(["/sbin/route", "-n", "get", "default"],
                           capture_output=True, text=True, check=True)
    interface = next((line.split(":", 1)[1].strip() for line in route.stdout.splitlines()
                      if line.strip().startswith("interface:")), None)
    if interface:
        result = subprocess.run(["/usr/sbin/ipconfig", "getifaddr", interface],
                                capture_output=True, text=True)
        address = result.stdout.strip()
        if result.returncode == 0 and address and not address.startswith("127."):
            return address
    raise RuntimeError("Cannot find the Mac LAN address. Set AOO_DEV_HOST to its Wi-Fi IP.")

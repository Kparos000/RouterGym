# Network Connectivity Troubleshooting (Workstations)

**Category:** Hardware  

## Description
Wired or wireless network issues from a single user’s device (not site-wide outages).

## Typical Ticket Patterns
- “Cannot connect to office Wi-Fi”
- “Network cable unplugged” warning
- “VPN connects but no internet”

## Resolution Steps
1. Check if other users or devices on the same network segment are affected.
2. Confirm Wi-Fi is enabled or network cable is properly seated and undamaged.
3. Run basic tests: `ipconfig`/`ping` (or equivalents), check for valid IP and gateway.
4. Forget and re-add Wi-Fi network or disable/enable network adapter.
5. Verify VPN client status and split-tunnel policies; reconnect if required.
6. Update network adapter drivers and restart the device.
7. If only this device fails on a known-good network, assign for onsite diagnosis.

## When to Escalate
- Multiple users in the same area report connectivity loss.
- Suspected switch, router, or firewall outage.
- VPN or proxy issues impacting a full site/region.

## Related Articles
- VPN Access Troubleshooting
- Remote Access RDP Troubleshooting
- Storage Quota Management



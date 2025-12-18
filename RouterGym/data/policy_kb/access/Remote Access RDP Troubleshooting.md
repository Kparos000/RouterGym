# Remote Access RDP Troubleshooting

**Category:** Access  

## Description
Users cannot connect to remote desktops or servers via RDP or similar tools.

## Typical Ticket Patterns
- “RDP connection failed”
- “Remote desktop black screen”
- “Cannot reach office PC from home”

## Resolution Steps
1. Confirm RDP target is powered on and reachable on the network/VPN.
2. Verify user is in the correct remote access/desktop groups.
3. Check firewall rules and RDP configuration on target machine.
4. Ask user to test from another device or network if feasible.
5. Review logs for failed logins or network errors; update client as needed.

## When to Escalate
- Many users cannot reach the same host or farm.
- Critical servers affected.
- Security concerns (excessive failed logins or suspected brute-force).

## Related Articles
- VPN Access Troubleshooting
- Access Request Flow (New, Modify, Remove)
- Administrative Rights Request Flow



# VPN Access Troubleshooting

**Category:** Access  

## Description
Problems connecting to corporate VPN or accessing internal resources over VPN.

## Typical Ticket Patterns
- “VPN client stuck on connecting”
- “Connected but cannot reach internal sites”
- “VPN login failed – certificate error”

## Resolution Steps
1. Confirm user has VPN entitlement in access system.
2. Check network connectivity (home internet, Wi-Fi quality).
3. Review VPN client logs for common errors (credentials, certificate, tunnel).
4. Ask user to restart device and try a different network if possible.
5. Update VPN client and root certificates where required.
6. For split tunnel setups, verify DNS and routing configuration.

## When to Escalate
- Many users unable to connect to VPN.
- Site-to-site or firewall issues suspected.
- Access to regulated or high-risk environments.

## Related Articles
- Remote Access RDP Troubleshooting
- Network Connectivity Troubleshooting (Workstations)
- Two-Factor Authentication (2FA) Setup and Issues



# Account Lockout Troubleshooting

**Category:** Access  

## Description
Resolving lockouts caused by repeated failed logins or cached credentials.

## Typical Ticket Patterns
- “Account locked every morning”
- “Too many failed login attempts”
- “Cannot unlock with self-service”

## Resolution Steps
1. Verify user identity and confirm which account is locked.
2. Unlock the account in AD or target system.
3. Identify possible sources of repeated failures (mobile devices, mapped drives, old sessions, VPN).
4. Ask user to update saved credentials on all devices and restart sessions.
5. Monitor for re-lock within 24 hours and advise user to change password if suspicion remains.

## When to Escalate
- Multiple accounts locked simultaneously (possible attack).
- User suspects credentials were exposed.
- Lockouts persist after full credential refresh.

## Related Articles
- Password Reset Procedure
- SSO Login Problems
- VPN Access Troubleshooting



# Two-Factor Authentication (2FA) Setup and Issues

**Category:** Access  

## Description
Enrolling users in 2FA (e.g., authenticator app, SMS, hardware token) and fixing related issues.

## Typical Ticket Patterns
- “New phone, lost 2FA codes”
- “Authenticator app not working”
- “2FA challenge not received”

## Resolution Steps
1. Verify user identity using backup verification methods.
2. Check their 2FA status in the identity provider or security portal.
3. Reset 2FA and guide user through re-enrollment on their device.
4. Confirm time sync on mobile device if using TOTP apps.
5. Validate login to at least one key application after setup.

## When to Escalate
- Repeated 2FA failures or suspected man-in-the-middle attacks.
- Missing or malfunctioning hardware tokens for critical users.
- IdP-wide 2FA delivery problems (e.g., SMS provider outage).

## Related Articles
- Password Reset Procedure
- SSO Login Problems
- VPN Access Troubleshooting



# SSO Login Problems

**Category:** Access  

## Description
Issues logging into applications that use single sign-on (SSO) and corporate identity provider.

## Typical Ticket Patterns
- “SSO loop when logging into app”
- “Stuck on sign-in page”
- “SSO works for email but not for Confluence”

## Resolution Steps
1. Confirm user can log into primary SSO portal.
2. Check if the specific application is assigned to the user in the SSO admin console.
3. Ask user to clear browser cache, try another browser, or use incognito.
4. Review error codes or SSO logs; verify attributes (email, username, groups) match app expectations.
5. Re-sync user or groups if mapping is incorrect; test again.

## When to Escalate
- Many users affected for the same app.
- IdP or federation outage suspected.
- Applications used for business-critical workflows.

## Related Articles
- Two-Factor Authentication (2FA) Setup and Issues
- Access Request Flow (New, Modify, Remove)
- HR Portal Troubleshooting



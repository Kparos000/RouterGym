# File Permissions Management

**Category:** Storage  

## Description
Correcting and maintaining file/folder permissions on shared storage.

## Typical Ticket Patterns
- “User can see folder but cannot save”
- “Need read-only access for external vendor”
- “Can’t remove someone from project folder”

## Resolution Steps
1. Confirm path, parent folder, and business owner.
2. Review current ACLs or permission groups.
3. Apply least-privilege changes (add/remove specific rights or groups).
4. Ensure inheritance settings are correct to avoid breaking other folders.
5. Ask user to re-test from their normal workstation.

## When to Escalate
- Large nested structures with complex inheritance.
- Requests impacting external or cross-company access.
- Accidental exposure of sensitive data.

## Related Articles
- Shared Drive Access Procedure
- Shared Folder Access Guide
- HR Data Privacy and Sensitive Information Requests



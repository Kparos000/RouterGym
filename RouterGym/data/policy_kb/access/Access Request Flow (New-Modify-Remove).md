# Access Request Flow (New, Modify, Remove)

**Category:** Access  

## Description
Standard workflow for creating, changing, or removing user access to systems (e.g., Git, Jira, Confluence, line-of-business apps).

## Typical Ticket Patterns
- “Need access to Confluence / Git repo”
- “Please add user to project group”
- “Remove access for leaver”

## Resolution Steps
1. Verify requester identity and approval (manager or system owner).
2. Identify exact system, group, or repository required and role level.
3. Check for existing access to avoid duplicates.
4. Apply change via IAM tool, AD group, or application admin console.
5. Ask user to log off and back in to refresh permissions.
6. Confirm access works for key actions; update ticket with groups/roles applied.

## When to Escalate
- Requests for elevated or production access.
- Conflicting approvals or unclear ownership.
- Bulk access changes for projects or migrations.

## Related Articles
- User Permissions Overview
- Administrative Rights Request Flow
- HR System Access Guide (HRIS / Oracle)



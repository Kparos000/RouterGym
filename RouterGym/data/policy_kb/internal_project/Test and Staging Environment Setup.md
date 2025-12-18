# Test and Staging Environment Setup

**Category:** Internal Project  

## Description
Creating and maintaining non-production environments for testing and UAT.

## Typical Ticket Patterns
- “Need staging environment cloned from prod”
- “UAT environment not in sync”
- “Test DB refresh request”

## Resolution Steps
1. Clarify environment purpose (dev, QA, UAT, performance).
2. Define data requirements (anonymised vs production clone) and security controls.
3. Provision infrastructure and apply configuration matching production as closely as allowed.
4. Set up regular refresh schedule and access controls.
5. Document environment endpoints and access instructions for the team.

## When to Escalate
- Need for production data in test environments.
- High performance or load-testing requirements.
- Complex multi-system integration environments.

## Related Articles
- Project IT Resource Requirements
- Data Access and Security for Project Teams
- System Deployment Checklist



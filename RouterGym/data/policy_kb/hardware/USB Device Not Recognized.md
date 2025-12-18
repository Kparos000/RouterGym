# USB Device Not Recognized

**Category:** Hardware  

## Description
System fails to detect or mount USB drives, smartcards, cameras, or other USB devices.

## Typical Ticket Patterns
- “USB stick not showing up”
- “Unknown USB device” warnings
- “Smartcard not detected”

## Resolution Steps
1. Test the USB device on another known-good machine if possible.
2. On the user’s device, try a different USB port and check Device Manager/System Info for errors.
3. Remove ghost/unrecognized USB entries and rescan for hardware changes.
4. Confirm corporate policy allows the device type (e.g., external storage may be blocked).
5. If allowed but blocked, check security software logs and apply exceptions as per policy.
6. For smartcards, ensure correct middleware/driver is installed and up-to-date.
7. Document the final working state or reason for blocking.

## When to Escalate
- Corporate security tools are blocking a business-critical device.
- Data recovery is required from a failing USB device.
- Multiple users report USB issues after a recent OS update.

## Related Articles
- File Permissions Management
- Hardware Replacement Process
- Privileged Account Usage Guidelines



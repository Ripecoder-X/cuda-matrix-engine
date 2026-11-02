# Handoff Notes

One file per batch: `handoffs/<BATCH_ID>.md`. Use the template below exactly. Do not paste source
code into a note; reference file paths and line counts.

```markdown
# Handoff — Batch <ID>: <title>

- **Agent / session:** <name or identifier>
- **Role used:** <ARCH | CORE | RUNTIME | KERNEL | API | TEST | BENCH | OPT | INTEG | VERIFY>
- **Date:** <YYYY-MM-DD>
- **Status at end of session:** <UNDER_REVIEW | BLOCKED | IN_PROGRESS (checkpoint)>

## Done
- <bullet list of completed items, mapped to the batch's acceptance criteria>

## Not done
- <bullet list, or "none">

## Files changed
- <path> — created / modified — <one-line reason>

## Decisions made (and where recorded)
- <decision> — recorded in <docs/DESIGN_DECISIONS.md section or "not recorded: reason">

## Verification
- Command: `<exact command>`
- Result: <pass/fail, test counts, key output line(s)>

## Blockers / questions for owner
- <or "none">

## Next step (exact command or action for the next agent)
- <one concrete action>
```

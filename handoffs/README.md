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

## Batch index

Notes exist for batches 0.1 through 7.3. Batches 8.1 through 8.3 have no note. Their status on
`TASK_BOARD.md` was set by owner instruction, not by a recorded verification run.

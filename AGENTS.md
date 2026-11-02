## 0. Read this first (for every agent)

You are one agent in a chain. Agents may run out of context, time, or quota at any moment, and the next agent will pick up where you stopped. This file is designed so that **any agent can resume the project from any point** without asking the owner to re-explain anything.

Follow this loop every session:

1. **Read this file** sections 0, 1, 2, 3 and 4 in full. Do not skip the interface contracts in section 4. They are binding.
2. **Read `TASK_BOARD.md`** (created in Batch 0.3). Find the first batch whose status is not `VERIFIED`.
3. **Read the batch** in section 6 that matches that ID. Check its dependencies are `VERIFIED`. If a dependency is not verified, stop and mark your batch `BLOCKED` with the reason.
4. **Mark the batch `IN_PROGRESS`** on `TASK_BOARD.md` before writing any code. Record your agent name and the date.
5. **Do only the files listed for that batch.** If you need a change in a file owned by another batch, do not make it. Write the request in the batch's handoff note and mark yourself `BLOCKED`.
6. **Run the batch's verification command.** Paste the actual output summary into the handoff note.
7. **Write a handoff note** to `handoffs/<BATCH_ID>.md` using the template in section 8.
8. **Update `TASK_BOARD.md`** to `UNDER_REVIEW` (you finished it) or `BLOCKED`. Only the verifier role moves a batch to `VERIFIED`.
9. **Stop.** Do not start the next batch in the same session unless your context clearly has room and the next batch is small. Finishing one batch cleanly is worth more than starting two and leaving a mess.

### Non-negotiable rules

- **Interfaces are frozen.** Section 4 defines every public signature. Do not rename, reorder, or change types. If you believe an interface is wrong, mark `BLOCKED` and explain. Do not silently change it.
- **Public headers in `include/cme/` must not include any CUDA header.** Host applications must compile without CUDA headers. All CUDA-dependent code lives in `src/cuda/`.
- **A file's status is not a batch's status.** "It compiles" is not done. A batch is done only when its verification command passes and its acceptance criteria are all checked.
- **Never claim performance you have not measured.** Never claim the project beats cuBLAS unless a benchmark in `results/` shows it, with hardware recorded.
- **Never make up numbers, versions, or profiler output.** If you do not know a value, run the command that produces it. If you cannot run it, write `NOT MEASURED` and say why.
- **Do not add features outside scope.** Scope is FP32 GEMM, row-major, no transposes, no Tensor Cores, synchronous public API. Anything else goes in `docs/DESIGN_DECISIONS.md` under "Deferred", not into code.
- **Keep your session output small.** Do not paste whole source files into notes. Reference file paths and line counts.

### If your context is running out

Write a checkpoint immediately, before you run out:

- Update `TASK_BOARD.md` to `IN_PROGRESS` with a one-line "next step" for the batch.
- Write `handoffs/<BATCH_ID>.md` with what is done, what is not, and the exact next command to run.
- Leave the repository in a building state if at all possible. If it cannot build, say so in the handoff note at the top.

---

If you are a free-tier or limited-context agent, do one batch per session.

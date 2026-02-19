# Delegating Work to the Codex Agent via tmux

## Session Setup
- The codex agent runs in a tmux session named `codex`
- It runs OpenAI Codex CLI (`codex` command) in the project working directory
- The agent is highly capable and can handle complex, long-running tasks given precise instructions
- Do NOT underestimate it — delegate aggressively, not conservatively

## Understanding the Agent's Behavior
- Treat the agent like it is extremely literal-minded: it follows instructions with exact precision, no more, no less
- It will work tirelessly for hours on a task without complaint — give it the full scope, not just a small piece
- It only stops when your instructions are ambiguous or insufficient for it to continue
- If it stops or asks a question, that means YOUR instructions were unclear — fix the instructions, not the agent
- Ambiguity is the enemy: every detail you leave unspecified is a potential point where the agent will halt or guess wrong
- Give it zero-ambiguity instructions with complete context and it will deliver
- The agent excels at execution — writing code, running tests, committing — but not at planning or design decisions
- YOU are the planner and architect; the agent is the executor. Plan thoroughly, then delegate the full implementation

## Sending Messages

### The Golden Rule: Always Send Enter Separately
```bash
# Step 1: Send the text
tmux send-keys -t codex "Your message here"
# Step 2: ALWAYS send Enter as a SEPARATE Bash tool call
tmux send-keys -t codex Enter
```

**NEVER** chain Enter in the same command with `&&`. It gets lost.
**NEVER** include Enter as part of the send-keys text argument.
**ALWAYS** make Enter its own standalone Bash tool call.

### Cancelling a Prompt First
When the agent is waiting at a Yes/No prompt:
```bash
# Call 1: Cancel the prompt
tmux send-keys -t codex Escape
# Call 2: Wait for it to process (separate Bash call)
sleep 2
# Call 3: Send your new message
tmux send-keys -t codex "Your correction here"
# Call 4: Submit it
tmux send-keys -t codex Enter
```

### Reading Agent Output
```bash
# See current screen
tmux capture-pane -t codex -p | tail -50

# Don't sleep+check in one command if user might interrupt
# Instead, check immediately or use short sleeps
```

## Priming the Agent

### After /clear or Starting a New Task
When starting fresh (after `/clear` or a new session), the agent has no project context. Always prime it:
1. Tell it to read `AGENTS.md` first — this contains project conventions, architecture, testing rules
2. List the specific files relevant to the task it should read
3. Tell it to skip ALL `bd`/beads commands — they are not available in its environment
4. Tell it not to modify existing test files in `tests/`
5. Only THEN give the task instructions

### Example Priming Message
```
Read these files FIRST before doing anything:
1. AGENTS.md — project conventions, architecture, testing rules
2. <file1> — <why it's relevant>
3. <file2> — <why it's relevant>

After reading all files, implement the following fix/feature.
<detailed instructions>

Skip ALL bd/beads commands — they are not available in your environment.
Do NOT modify any existing test files in tests/.
```

## Giving Instructions

### Be Explicit and Comprehensive
The agent works best with:
- Exact file paths to read first
- Exact commands to run for verification
- Clear sequence of tasks
- What NOT to do (e.g., "Skip ALL bd/beads commands")
- Enforce TDD: "Write tests first, verify they fail, then implement"
- Tell it existing tests are acceptance tests that must not be modified
- Tell it to add new test files instead of modifying existing ones

### First Message Template
```
Read <spec-file> FIRST before doing anything. It is your single source of truth.
Follow it to the letter. Do not improvise. Do not deviate.
Execute Phase 1 through Phase N in order.
After EACH phase, run: <verification command>
After ALL phases, run: <final verification>
Commit after each phase as specified.
Skip ALL bd/beads commands — they are not available in your environment.
Do NOT modify existing test files in tests/ — add new test files instead.
If you cannot run a command due to sandbox restrictions, tell me and I will run it for you.
Important files to read: (1) ... (2) ... (3) ...
START NOW by reading the spec file.
```

### Correcting the Agent Mid-Task
- Cancel current prompt with Escape first
- Be direct: "STOP. Do X instead of Y."
- Tell it what went wrong and what to do differently

## Agent Limitations

### What It Cannot Do
- Run `bd`/beads commands (not in PATH, missing BEADS_DIR env var)
- Run tmux-based TUI tests (sandbox restriction on tmux)
- Access authenticated services
- Run `just ci-quiet` (may hit sandbox restrictions)

### What I Must Handle
- All beads ticket operations (create, close, sync)
- Running TUI integration tests and reporting results back
- Fixing go.mod/go.sum issues (go mod tidy)
- Final CI validation (just ci-quiet) — ALWAYS run this, never skip

### When It Gets Stuck
- Permission errors on files → tell it to create new files instead of modifying
- Sandbox restrictions → offer to run commands for it and report results
- Missing tools → tell it to skip and you'll handle it

## Review Protocol

### ALWAYS Validate with CI
After the agent finishes ALL work, ALWAYS run:
```bash
just ci-quiet
```
This is the definitive validation. Do NOT rely on `go test ./...` alone — it misses:
- Format checks (`go fmt`)
- Lint checks
- CLI integration tests (shell scripts in tests/)
- TUI integration tests
- Build verification

If `just ci-quiet` fails, investigate the failure. Common causes:
- New files shift TUI cursor positions in existing keystroke-based tests
- go.mod/go.sum need `go mod tidy`
- New test files missing `chmod +x`

### After Each Phase
- Check git log to verify commits
- Read the code it produced
- Run `just ci-quiet` (not just `go test`)
- Only close beads tickets after CI passes

### Quality Checks
- Does the code match the spec?
- Are architecture rules respected? (no forbidden imports)
- Does `just ci-quiet` pass? (the ONLY check that matters)
- Are test files executable? (`chmod +x`)
- Is go.mod clean? (`go mod tidy`)
- Do new features cause regressions in existing tests?

## Anti-Patterns (Mistakes I Made)

1. **Being too conservative** — The agent can handle complex work. Delegate the full feature, not just the foundation.
2. **Guessing test outcomes** — Never claim a test will pass or fail without running it. Run `just ci-quiet` to verify.
3. **Forgetting Enter** — Always send Enter as a separate Bash call.
4. **Not enforcing TDD** — Explicitly tell the agent to write tests first, verify they fail, then implement.
5. **Handling tickets the agent could skip** — Just tell it to skip beads upfront rather than letting it fail.
6. **Not telling it about existing test files** — Tell it existing tests are acceptance tests that must not be modified.
7. **Running `go test` instead of `just ci-quiet`** — `go test` only covers unit tests. `just ci-quiet` is the full CI pipeline including format, lint, CLI tests, TUI tests, and build. Always use `just ci-quiet` for final validation.
8. **Not checking for regressions in TUI tests** — New files created by the feature (e.g., `.guardignore` on init) can shift cursor positions in keystroke-based TUI tests. Always run full CI to catch these.

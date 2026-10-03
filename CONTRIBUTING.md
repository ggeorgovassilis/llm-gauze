# Contributing

Thanks for your interest in contributing to llm-gauze. This guide covers the
*process* — how issues and pull requests flow through the project. For how to
run the gateway, run the tests, and add a detector or remediation, see
[`DEVELOPING.md`](DEVELOPING.md).

## Reporting issues

- Search the [open issues](https://github.com/ggeorgovassilis/llm-gauze/issues)
  first to avoid duplicates.
- Use a descriptive title and explain what you expected, what actually happened,
  and the steps to reproduce it.
- Issues are triaged into one of two milestones:

  - **`current`** — planned for the next release.
  - **`backlog`** — accepted but not yet scheduled.

- If the bug can be reproduced with a failing test, that test is the best
  possible issue description — see [`DEVELOPING.md`](DEVELOPING.md).

## Development workflow

Every change starts from a ticket and is reviewed before it lands:

1. **Pick or file a ticket.** Every piece of work has a corresponding issue.
   If there isn't one yet, create it first.
2. **Create a branch per ticket.** Branch off `main` with a short name that
   references the ticket, e.g. `20-license` or `57-coverage-gaps`.
3. **Make the change and test it.** Add or update tests for any behaviour you
   change, and run the full suite:

   ```bash
   ./scripts/test.sh
   ```

   The suite must be green before you commit — never leave a behaviour change
   without test coverage.
4. **Commit with a concise message** that references the ticket, e.g.
   `Add MIT license (#20)`. One logical change per commit.
5. **Push and open a pull request.** Link the ticket with `Closes #<n>` in the
   description so it is closed automatically on merge. CI runs
   `ruff check`, `ruff format --check`, `mypy`, and `pytest` on every pull
   request.

## Pull request review

- Pull requests are squash-merged into `main`; a clean, self-contained history
  is preferred over many small commits.
- A maintainer reviews the change. Address review comments in the same branch.
- Once the change is approved, the maintainer merges the pull request and the
  linked ticket is closed.

## Commit conventions

- Concise imperative subject line, referencing the ticket: `Add …`, `Fix …`,
  `Remove …`.
- Keep each commit to one logical change; split unrelated work into separate
  branches.
- New tests live in `tests/` as `test_*.py` with module-level `test_*`
  functions (no per-file `__main__` runners).

## Licence

By contributing you agree that your contributions are licensed under the same
[MIT licence](LICENSE) as the project.

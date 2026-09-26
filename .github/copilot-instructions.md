# Overview

This project is an LLM gateway that detects and remediates shortcommings of local LLMs.

# Talking with the user

- Project language is UK english
- Be brief, short, to the point
- Explore one option at a time
- When the user asks you a question, just answer the question, don't undertake any modifications.
- When asked to come up with a plan, formulate the plan in self-contained phases. Each phase delivers one thing and has clear acceptance criteria.
- When the user asks you to "commit", they mean commit to github (with a concise message) and push.
- When the user asks you to list tickets, use the `gh` tool to list open repository issues. Group them by milestone (`current` first, `backlog` last). Make the ticket ID clickable with a link to the github issue.

# Project tech
- Python in docker
- Pin library/framework dependencies
- Write wrapper scripts
- The github cli `gh` is installed and authenticated to the repository.

# Project structure
- The project is a github repository `bandaid`.
- `docs/` - Documentation
- `source/` - Source code
- `tests/` - Tests
- `scripts/` - Wrapper scripts

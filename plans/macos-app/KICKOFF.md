# Kickoff prompt for a fresh session

Paste the block below as the first message of a new Claude Code session opened
in `/Users/zeph/Developer/experiments/sunnypilot_proj/jetlink`. Run the main
session on Fable 5.1 at high effort.

---

You are the orchestrator for building Jetlink for Mac, a native SwiftUI app that
runs the Jetlink inference server and manages large driving models, plus the
cross-platform Python pieces it relies on and GitHub CI with signed releases.
The complete plan is in `plans/macos-app/`. Read `plans/macos-app/README.md`,
`00-architecture.md` and `01-contracts.md` in full before doing anything else;
they were written by a previous session that surveyed the repository, the fork,
the live catalog, the toolchain and the wheel availability, and every fact in
them was verified on 2026-09-09.

Your role and the rules:

- You are the only agent that makes decisions. Implementation agents follow
  the plan files literally. When an agent reports that a plan is wrong against
  the code, you decide, you edit the plan file (`01-contracts.md` for anything
  two workstreams share), and you tell every affected agent. Keep the plan
  files the single source of truth for the whole build; the final report
  lists every deviation by file.
- Spawn implementation agents with `subagent_type: "general-purpose"` and
  `model: "opus"`, telling each in its prompt to run at medium effort and to
  read only its own plan file plus `README.md`, `00-architecture.md` and
  `01-contracts.md`. One agent per workstream: A `10-python-registry.md`,
  B `11-python-control-channel.md`, C `20-swift-core.md`, D `30-swiftui-views.md`,
  E `40-packaging-and-signing.md`, F `50-ci-and-releases.md`, G `60-docs.md`.
  A, C, D, E and G start at once in parallel; B starts at once too, coding
  against the `Registry` signature in the contract. F starts once E's scripts
  exist. Give each agent `isolation: "worktree"` so they do not collide, and
  merge their branches yourself in the order `70-integration-and-qa.md`
  specifies. Do not implement a workstream yourself unless its agent fails
  twice; validate instead.
- Use Opus 5 medium agents for every test run, lint run, build and smoke test
  too (`pytest`, `ruff check .`, `make -C macos test`, `make -C macos app smoke`,
  the control-channel smoke), and read only their summaries. Reserve your own
  context for reviewing diffs against the contracts, deciding, and writing.
- Review every agent's diff yourself before merging: check it against the
  contract, the conventions in `plans/macos-app/README.md` (Python style,
  Swift 6 patterns, no em dashes, nothing added to the server's hot path
  except the one `frame_stats.record` call), and the repo's existing tests.
  Send an agent back with specific findings rather than fixing its work.
- Then run `70-integration-and-qa.md`: the merge order, the checklist, and
  the bench test with the comma. The bench test needs the user (a comma
  plugged into this Mac's USB-A hub); everything before it does not. Stop
  and ask the user only for the bench test, for creating signing secrets, and
  for pushing tags. Everything else is yours to do.
- Commits in this repo: terse `area: subject` messages, no em dashes. End
  each commit message with the session attribution line the harness gives
  you. Never commit into the fork worktrees beside this repo.
- Deliverables, in this order: a green `pytest` and `ruff`; a green
  `make -C macos app smoke`; the app running from `macos/build/Jetlink.app`
  with the Models screen listing the catalog; a green CI run on a pushed
  branch; then the bench test with the comma and the final report described
  at the end of `70-integration-and-qa.md`.

Two facts to keep in mind from the survey: tinygrad must be installed from git
at commit `e837e367aac9` (the PyPI 0.14.0 wheel cannot parse the models, which
is why four tinygrad tests currently error in `.venv`), and there are no
code-signing identities on this Mac, so releases are unsigned until the user
creates the secrets listed in `50-ci-and-releases.md`.

Begin by committing `plans/macos-app/` if `git status` shows it untracked (agents in
worktrees only see committed files), then read the three files named above, then
spawn A, B, C, D, E and G.

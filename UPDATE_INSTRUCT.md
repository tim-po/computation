# UPDATE_INSTRUCT — Agent Instructions for Document Management

Instructions for any agent (human or AI) updating this document repository.

---

## Repository structure

```
docs/
  STREAMS.md               ← overview of LAN vs WAN streams + viability matrix
  INDEX.md                 ← master registry, organized by stream
  research/
    lan/                   ← 🟢 current focus (RTT ~5ms)
    wan/                   ← 🔵 future scaling (RTT ~100ms)
  architecture/            ← cross-cutting architectural decisions and postmortems
  updates/                 ← cofounder briefs and time-stamped updates (immutable)
  diagrams/                ← .mmd source files and rendered .png exports
  drafts/                  ← work-in-progress, not yet promoted to a stream
UPDATE_INSTRUCT.md         ← this file
```

---

## Research streams

The project has two streams, differentiated by RTT between orchestrator and nodes:

| Stream | RTT | Folder | Status |
|---|---|---|---|
| **LAN** | ~5ms | `research/lan/` | 🟢 Current focus |
| **WAN** | ~100ms | `research/wan/` | 🔵 Future scaling |

When adding a research document, place it in the appropriate stream folder. If a document is relevant to both streams, place it in `architecture/` and set `stream: both`.

See [`docs/STREAMS.md`](docs/STREAMS.md) for the full architecture viability matrix (which approaches work on LAN vs WAN).

---

## Frontmatter format

Every `.md` document in `docs/` must have YAML frontmatter:

```yaml
---
title: "Human-readable title"
date: YYYY-MM-DD          # date created or last substantively updated
author: Name or TBD
version: vX.Y             # bump minor (v1.0 → v1.1) for content changes, major (v1.x → v2.0) for structural rewrites
status: draft | research-draft | stable | living-document | final | archived
stream: lan | wan | both   # required for research/ and architecture/ docs
tags: [tag1, tag2, ...]   # lowercase, hyphenated; include the stream as a tag too
---
```

**Status meanings:**
- `draft` — early idea, unstable
- `research-draft` — formal spec under active investigation
- `stable` — decision finalized, not expected to change
- `living-document` — updated continuously as the project evolves
- `final` — immutable snapshot (briefs and updates)
- `archived` — superseded; keep the file, add `superseded_by:` field

---

## Versioning rules

| Change type | Version bump | Example |
|---|---|---|
| Typo fix, formatting | none | — |
| Minor content addition or clarification | minor | v1.0 → v1.1 |
| New section or significant rewrite | minor | v1.2 → v1.3 |
| Breaking restructure or complete rewrite | major | v1.x → v2.0 |

Always update the `date:` field in frontmatter when bumping the version.

---

## Adding a new document

1. Choose the correct folder:
   - **research/lan/** — hypothesis, experiment, or spec scoped to LAN (RTT ~5ms)
   - **research/wan/** — hypothesis, experiment, or spec scoped to WAN (RTT ~100ms)
   - **architecture/** — cross-cutting design decisions, postmortems, living docs (set `stream: both` if spans both)
   - **updates/** — cofounder brief or time-stamped progress update
   - **drafts/** — not ready to be categorized yet

2. Name the file descriptively. Use kebab-case. For updates, include the date: `cofounder_update_YYYY-MM-DD.md`.

3. Add YAML frontmatter (see format above).

4. Add the document to `docs/INDEX.md` in the correct table section. One row per document.

5. If the document references diagrams, place diagram files in `docs/diagrams/` and use relative paths: `../diagrams/filename.png`.

---

## Updating an existing document

1. Update content.
2. Bump `version:` (minor or major depending on change size).
3. Update `date:` to today.
4. If status changed, update `status:`.
5. Update the corresponding row in `docs/INDEX.md` (version and status columns).

**Exception: `updates/` documents are immutable.** Cofounder briefs and updates are snapshots in time — never edit them after publishing. If a correction is needed, create a new update document and reference the original.

---

## Creating a draft

1. Place the file in `docs/drafts/`.
2. Use status `draft` in frontmatter.
3. Add to the Drafts section of `docs/INDEX.md`.

### Promoting a draft

When a draft is ready:
1. Move the file to the appropriate folder (`research/`, `architecture/`, etc.).
2. Update frontmatter: change `status:` to the correct value.
3. Remove from Drafts table in `docs/INDEX.md` and add to the correct table.

---

## Writing a cofounder update

Updates are snapshots, not living documents.

**Filename:** `cofounder_update_YYYY-MM-DD.md` (add a short slug if multiple updates on the same day: `cofounder_update_YYYY-MM-DD_skeleton.md`)

**Required frontmatter fields:**
```yaml
---
title: "Update: <short description>"
date: YYYY-MM-DD
author: <your name>
version: v1.0
status: final
tags: [update, <relevant tags>]
---
```

**Typical structure:**
- TL;DR / headline numbers
- Part 1: current results / what finished
- Part 2: new direction or idea (if any)
- Open questions / what you need from the other person
- Compute budget consumed today / remaining

After writing, add to the Updates table in `docs/INDEX.md`.

---

## Archiving a document

When a document is superseded:
1. Add `superseded_by: path/to/new-doc.md` to its frontmatter.
2. Change `status:` to `archived`.
3. Update `docs/INDEX.md`: move to an Archived section (create if needed), or remove.
4. Do not delete the file — history matters.

---

## Updating the INDEX

`docs/INDEX.md` is the registry. Keep it accurate:
- Every document in `docs/` must have a row.
- Every row must have correct version and status.
- Check the index after adding, moving, or archiving any document.

---

## Cross-document links

Use relative paths from the file's own location:

```markdown
# From docs/architecture/decentralized-llm-synthesis.md:
See `../research/dst-research-spec.md` for the full spec.

# From docs/research/dst-research-spec.md:
Fallback described in `../architecture/failed_architectures.md`.
```

When moving a file, grep for all references to it and update them.

---

## Diagrams

- Source files: `.mmd` (Mermaid) in `docs/diagrams/`
- Rendered exports: `.png` in `docs/diagrams/`
- Reference in markdown: `![Alt text](../diagrams/filename.png)`
- After editing a `.mmd`, regenerate the `.png` and commit both.

---

## Quick checklist for any document change

- [ ] Frontmatter present and correct (title, date, author, version, status, tags)
- [ ] Version bumped if content changed
- [ ] `docs/INDEX.md` updated
- [ ] Internal links use relative paths and are not broken
- [ ] Diagrams referenced exist in `docs/diagrams/`

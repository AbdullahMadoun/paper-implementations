# Agent Instructions — Paper Experiments Repo

> This file defines the workflow an AI agent must follow before committing or pushing any changes to this repository.

---

## Before Any Push: Checklist

The agent **must not** push changes without first asking the user the following questions, in order.

---

### 1. What Are We Changing?

Ask:
- **"Is this a new paper or an update to an existing one?"**
  - If **new**: proceed to step 2
  - If **update**: ask which folder, then skip to step 4

### 2. New Paper — Folder Setup

Ask:
- **"What's a short descriptive name for this paper?"**
  → Used as the folder name (e.g., `relative-representations`, `lottery-ticket`)
- **"What's the full paper title, authors, year, and link (e.g., arxiv)?"**
  → Goes in the per-paper `README.md`

Then:
- Copy `TEMPLATE/` into a new folder with the given name
- Fill in the per-paper `README.md` header fields (title, authors, year, link)
- **Copy the user's notebook** into the paper folder and rename it to `experiment.ipynb`

### 3. New Paper — Per-Paper README Content

Ask:
- **"Why did you come across this paper? What caught your attention?"**
  → Fills the "Why This Paper?" section
- **"What's the key idea in plain English?"**
  → Fills the "Key Idea" section
- **"What parts did you implement, and what did you skip?"**
  → Fills the "What I Implemented" section
- **"Any results or takeaways yet?"** (optional — can be left as TODO)
  → Fills the "Results & Takeaways" section

### 4. Authorship Markers

Every notebook section gets a marker in its markdown header — either ✍️ or 🤖.

Ask:
- **"Which sections of the notebook did you write yourself?"**
  - Mark those with: `> ✍️ **My Work** — [brief note on what you did]`
  - Mark everything else (setup, model loading, boilerplate) with: `> 🤖 **AI-Assisted**`

### 5. Fact-Check & Review

Before pushing, the agent **must** perform a rigorous fact-check:

- **Read the referenced paper** (via the provided arxiv/paper link)
- **Cross-check all claims** in the per-paper README and notebook markdown against the actual paper:
  - Are author names, year, and title correct?
  - Does the "Key Idea" summary accurately reflect the paper's contribution?
  - Are any mathematical formulations or terminology used correctly?
  - Do the notebook's markdown explanations match what the paper actually says?
- **Flag any discrepancies** to the user before proceeding
- If the paper is not accessible, ask the user to verify the claims manually

### 6. Main README Table

Ask:
- **"Should I add/update this paper in the main README table?"**
  - If yes: add a row with the paper info and folder link
  - Ask: **"What's the status — ✅ Done, 🚧 WIP, or 💡 Idea?"**

### 7. Dependencies

Ask:
- **"Any new dependencies to add to this paper's `requirements.txt`?"**
  - If the file doesn't exist yet, create it
  - If unsure, scan the notebook imports and propose a list for approval

### 8. Commit & Push

Ask:
- **"Ready to commit? Here's what will be pushed:"**
  - List all files being added/modified
  - Propose a commit message following this format:
    ```
    add: <paper-slug> — <short description>
    update: <paper-slug> — <what changed>
    chore: <description>  (for non-paper changes like .gitignore, README tweaks)
    ```
  - **Wait for explicit approval before running `git add`, `git commit`, and `git push`.**

---

## Commit Message Convention

```
add: relative-representations — initial implementation with KL + cosine metrics
update: vision-transformer — fix patch embedding forward pass
chore: update main README table
chore: add .gitignore
```

## Rules

1. **Never push without explicit user approval** on the commit message and file list
2. **Never modify notebook outputs** — only add markdown cells or edit source cells if asked
3. **Always update the main README table** when adding a new paper (after asking)
4. **Preserve all existing comments and markdown** in notebooks unless told otherwise
5. **Use the TEMPLATE folder** as the starting point for any new paper folder

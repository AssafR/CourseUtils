# Git Instructor Manual (Full Edition)

# Overview

This lab accompanies the Git presentations and is intended for beginner AI Developers and programmers.

Recommended duration: 90–120 minutes

Learning outcomes:

Students will be able to:

- Create repositories
- Commit changes
- Inspect history
- Restore files
- Revert commits
- Create branches
- Merge branches
- Resolve merge conflicts
- Work with GitHub remotes
- Understand fetch vs pull
- Use Git for AI experiment tracking

---

# Exercise 0 – Setup (5–10 min)

## Instructor Demo

Show:

```bash
git --version
```

Explain:

Git is already installed on most development machines.
Today we will use:

- Terminal
- IDE (VS Code / PyCharm)
- GitHub

## Student Tasks

```bash
mkdir git_playground
cd git_playground
```

Checkpoint:
Everyone should now be inside the same working folder.

Common problems:

- Wrong folder
- Git not installed
- Terminal confusion

---

# Exercise 1 – First Repository (10 min)

## Goal

Understand:

Working Directory → Staging Area → Commit

## Live Demo

```bash
mkdir hello_git
cd hello_git

git init
```

Discuss:

What changed?

Show:

```bash
git status
```

Explain:

No commits yet.

## Student Tasks

Create:

```text
app.py
```

In IDE:

```python
print("Hello Git")
```

Run:

```bash
git status
```

Question:

Why is the file untracked?

Continue:

```bash
git add app.py
git status
```

Ask:

What changed?

Commit:

```bash
git commit -m "Add hello world script"
```

View history:

```bash
git log --oneline
```

Expected outcome:

One commit appears.

---

# Exercise 2 – Modifying Files (10 min)

## Goal

Understand:

- status
- diff
- commit history

Modify:

```python
print("Version 2")
```

Run:

```bash
git status
git diff
```

Discuss:

Git stores differences.

Commit:

```bash
git add app.py
git commit -m "Update app"
```

Show:

```bash
git log --oneline
```

Expected:

Two commits.

---

# Exercise 3 – Restore and Revert (10 min)

## Goal

Git as a time machine.

Break the file:

```python
this is invalid python
```

Run:

```bash
git diff
```

Restore:

```bash
git restore app.py
```

Explain:

Restore replaces local changes with last committed version.

### Safe Undo

Create another commit.

Then:

```bash
git revert HEAD
```

Discuss:

Revert creates a new commit.

Why companies prefer revert.

---

# Exercise 4 – Branches (10 min)

## Goal

Understand branch isolation.

Create:

```bash
git switch -c feature-new-message
```

Modify:

```python
print("Feature branch")
```

Commit.

Return:

```bash
git switch main
```

Question:

Where did the feature go?

Answer:

It exists only on the branch.

---

# Exercise 5 – Merge (5 min)

Merge:

```bash
git merge feature-new-message
```

Explain:

Git combines histories.

Delete branch:

```bash
git branch -d feature-new-message
```

---

# Exercise 6 – GitHub Remote (15 min)

## Instructor Preparation

Students need GitHub accounts.

Create empty repository.

Copy URL.

Connect:

```bash
git remote add origin REPOSITORY_URL
```

Verify:

```bash
git remote -v
```

Push:

```bash
git push -u origin main
```

Checkpoint:

Students see code on GitHub.

---

# Exercise 7 – Clone (5 min)

Create second copy:

```bash
cd ..
git clone REPOSITORY_URL hello_git_clone
```

Explain:

This simulates a second developer.

---

# Exercise 8 – Fetch vs Pull (10 min)

## Repository A

Create commit.

Push.

## Repository B

Run:

```bash
git fetch
```

Question:

Did files change?

Answer:

No.

Show:

```bash
git log origin/main --oneline
```

Then:

```bash
git pull
```

Question:

What changed now?

Key lesson:

Fetch = download information

Pull = fetch + merge

---

# Exercise 9 – Merge Conflict (15 min)

## Goal

Normalize conflicts.

Repository A:

Create:

```python
print("A")
```

Commit.

Push.

Repository B:

Create same file:

```python
print("B")
```

Commit.

Push.

Observe failure.

Pull.

Open conflict markers.

Show:

```text
<<<<<<< HEAD
=======
>>>>>>> origin/main
```

Resolve manually.

Commit resolution.

Discuss:

Conflicts are normal.

---

# Exercise 10 – AI Experiment Tracking (10 min)

Create:

```yaml
model: baseline_mlp
learning_rate: 0.001
batch_size: 32
epochs: 10
```

Commit.

Create branch:

```bash
git switch -c exp-lr-1e-4
```

Modify:

```yaml
learning_rate: 0.0001
```

Commit.

Discussion:

Why is this useful?

Expected answers:

- Experiment history
- Reproducibility
- Team collaboration

---

# Exercise 11 – Tags (5 min)

Mark best experiment:

```bash
git tag best-lr-1e-4
```

View:

```bash
git tag
```

Discuss:

Real-world uses:

- Releases
- Best models
- Milestones

---

# Fast Finishers

1. Recover deleted file.
2. Create two branches and merge both.
3. Create intentional conflict.
4. Tag three versions.
5. Compare two branches with git diff.

---

# Comfort Checklist

Students can:

- git init
- git status
- git add
- git commit
- git log
- git diff
- git restore
- git revert
- git switch
- git merge
- git remote add
- git clone
- git fetch
- git pull
- git push
- git tag
- resolve merge conflicts

End with:

"If you can perform all of these tasks without panic, your Git skills are already ahead of many junior developers."

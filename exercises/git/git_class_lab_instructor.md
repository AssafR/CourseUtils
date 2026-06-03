# Git Class Lab Guide (Instructor Version)

A complete hands-on Git lab designed to accompany the Git presentations.

## Part A – Local Git Fundamentals

### Exercise 1 – Create Your First Repository

**Learning Goals**
- Initialize a Git repository
- Understand repository status
- Create the first commit

**Commands**

```bash
git init
git status
git add app.py
git commit -m "Add hello world script"
git log --oneline
```

**Expected Observations**
- File starts as untracked
- After staging it is ready to commit
- After commit the repo is clean

---

### Exercise 2 – Modifying Files

```bash
git status
git diff
git add app.py
git commit -m "Update app.py"
```

---

### Exercise 3 – Restore and Revert

```bash
git restore app.py
git revert HEAD
```

Older tutorials may use:

```bash
git checkout -- app.py
```

## Part B – Branching

### Exercise 4 – Create a Feature Branch

```bash
git switch -c feature-new-message
git add .
git commit -m "Add feature message"
git switch main
```

### Exercise 5 – Merge a Branch

```bash
git merge feature-new-message
git branch -d feature-new-message
```

### Exercise 6 – Resolve a Merge Conflict

```bash
git pull
git add conflict_demo.py
git commit -m "Resolve merge conflict"
```

## Part C – Working with Remotes

### Exercise 7 – Connect to GitHub

```bash
git remote add origin REPOSITORY_URL
git remote -v
git push -u origin main
```

### Exercise 8 – Clone a Repository

```bash
git clone REPOSITORY_URL hello_git_clone
```

### Exercise 9 – Fetch vs Pull

```bash
git fetch
git log origin/main --oneline
git pull
```

## Part D – AI Developer Workflow

### Exercise 10 – Version a Training Configuration

```yaml
model: baseline_mlp
learning_rate: 0.001
batch_size: 32
epochs: 10
```

### Exercise 11 – Experiment Branches

Create:

```text
exp-lr-1e-4
```

Modify:

```yaml
learning_rate: 0.0001
```

### Exercise 12 – Tags

```bash
git tag best-lr-1e-4
git tag
```

## Part E – Challenge Exercises

1. Recover a deleted file.
2. Create two feature branches and merge both.
3. Intentionally create a merge conflict and resolve it.
4. Create three experiment branches and tag the best version.

## Comfort Checklist

Students should be comfortable with:

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
- tags
- merge conflicts
- experiment branches

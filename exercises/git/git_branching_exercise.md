# Git Branching Exercise – Alternate Realities

## Goal

- Create a branch
- Switch between branches
- Make independent changes
- Observe that branches are isolated
- Merge a branch back into `main`
- (Optional) Create a merge conflict

## Scenario

Imagine you are experimenting with different machine learning models.

You already have a baseline implementation.

You want to try different ideas without breaking the stable version.

This is exactly why branches exist.

## Part 1 – Create the Initial Project

Create `model.py`:

```python
def model_name():
    return "Baseline Linear Separator"

print(model_name())
```

Commit it:

```bash
git add model.py
git commit -m "Add baseline model"
git push
```

## Part 2 – Create Two Experimental Branches

### Student A

```bash
git switch -c experiment-linear
```

### Student B

```bash
git switch -c experiment-svm
```

## Part 3 – Modify the Model

### Student A

```python
def model_name():
    return "Improved Linear Separator"

print(model_name())
```

```bash
git add model.py
git commit -m "Improve linear separator"
```

### Student B

```python
def model_name():
    return "SVM Classifier"

print(model_name())
```

```bash
git add model.py
git commit -m "Try SVM classifier"
```

## Part 4 – Observe Branch Isolation

```bash
git switch main
```

What does the program print?

```bash
git switch experiment-linear
```

What does the program print now?

```bash
git switch experiment-svm
```

What does the program print now?

## Discussion

Why didn't the changes affect `main`?

## Part 5 – Merge One Experiment

```bash
git switch main
git merge experiment-linear
```

Run the program again.

What is now shown?

## Optional Challenge – Create a Merge Conflict

```bash
git switch main
git merge experiment-svm
```

Did Git merge automatically?

If not:
- Examine the conflict markers
- Resolve the conflict manually
- Choose a final model name
- Complete the merge

## Reflection Questions

1. Why are branches useful?
2. What problem do branches solve?
3. What would happen if everyone edited `main` directly?
4. In your own words, what does `git merge` do?

## Commands Used

```bash
git switch -c branch-name
git switch branch-name
git branch
git merge branch-name
git status
git diff
```

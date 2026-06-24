
# Git Class Exercise Worksheet (Student Version)

This worksheet contains hands‑on Git exercises **without solutions**.  
Follow the instructions step by step, using your terminal and IDE.

---

## Block 0 – Setup & Ground Rules

1. Verify that Git is installed.
2. Create a new working directory for this class.
3. Navigate into your workspace.

Write down any issues you encounter.

---

## Block 1 – First Local Repository: Core Workflow

1. Create a new folder for a project.
2. Initialize it as a Git repository.
3. Create a new Python file inside it.
4. Check the repository status.
5. Stage the file.
6. Commit the change with a meaningful message.
7. View the commit history.

---

## Block 2 – Changing & Undoing Safely

1. Modify the file you created earlier.
2. Check the status and review the changes.
3. Stage and commit the updated file.
4. Introduce an intentional mistake into the file.
5. Explore how to compare this broken version with the committed version.
6. Restore the file to the last committed version.
7. Make another commit, then practice undoing that commit using a safe method.

---

## Block 3 – Branching & Merging: Alternate Realities

### Scenario

Imagine your team has a baseline machine learning model:

```python
def model_name():
    return "Baseline Linear Separator"

print(model_name())
```

You want to experiment without risking the stable version.

---

### Step 1 – Create an Experimental Branch

Create a new branch:

```bash
git switch -c experiment-linear
```

Modify the file:

```python
def model_name():
    return "Improved Linear Separator"

print(model_name())
```

Commit the change:

```bash
git add model.py
git commit -m "Improve linear separator"
```

---

### Step 2 – Return to Main

Switch back:

```bash
git switch main
```

### Question

What does the program print now?

Why?

---

### Step 3 – Compare Alternate Realities

Switch back to the experimental branch:

```bash
git switch experiment-linear
```

### Question

What does the program print now?

What changed?

---

### Step 4 – Merge the Experiment

Return to main:

```bash
git switch main
```

Merge the branch:

```bash
git merge experiment-linear
```

Run the program again.

### Question

What does it print now?

---

### Reflection

Answer in your own words:

1. Why didn't the experimental change affect `main` immediately?
2. What problem do branches solve?
3. Why might a team prefer branches over editing `main` directly?

---

## Block 4 – Working With Remotes (GitHub)

1. Create an empty repository on GitHub.
2. Connect your local repository to the GitHub remote.
3. Push your local commits to GitHub.
4. Clone the same repository into a new folder.
5. Compare both local copies and verify they match the remote.

---

## Block 5 – Fetch vs Pull

Using your two local copies:

1. In the first copy, add a new change and push it to GitHub.
2. In the second copy, check for updates without modifying your files.
3. Inspect the remote branch to see the new commit.
4. Bring the update into your working copy.
5. Review the difference between “fetch” and “pull”.

---

## Block 6 – Merge Conflict Practice

1. In copy A, create a new file and commit it.
2. Push the commit to GitHub.
3. In copy B, create a file with the **same name**, but different contents.
4. Commit the change and attempt to push.
5. Observe the error message.
6. Pull the remote changes to trigger a conflict.
7. Open the file and resolve the conflict manually.
8. Stage, commit, and push the resolved version.

---

## Block 7 – AI-Focused Exercise: Config Versioning & Experiment Tracking

1. Create a new project for ML experiments.
2. Add a configuration file defining model parameters.
3. Commit the baseline configuration.
4. Create a new branch for an experiment.
5. Modify configuration parameters for the experiment.
6. Commit the experiment configuration.
7. Tag the experiment’s commit to mark it as a “best model version”.
8. Switch back to the main branch.

---

## Block 8 – Confidence Checklist

Check off the Git concepts you feel comfortable with:

- [ ] Initializing repositories  
- [ ] Tracking and committing changes  
- [ ] Viewing history  
- [ ] Inspecting modifications  
- [ ] Undoing changes safely  
- [ ] Using branches  
- [ ] Merging branches  
- [ ] Connecting to remotes  
- [ ] Pushing and pulling  
- [ ] Resolving merge conflicts  
- [ ] Using tags for experiment tracking  

---

Good luck, and remember: **Git rewards curiosity and experimentation.**

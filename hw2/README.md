# HW2: branches and pull request

Practice of the basic GitHub workflow: work in a separate branch, open a pull request and merge it into `main`.

## Workflow

```bash
git switch -c feature/hw2        # create a branch from main and switch to it
# ... edit files ...
git add -A
git commit -m "Add HW2 description"
git push -u origin feature/hw2   # publish the branch
gh pr create --base main --head feature/hw2 \
  --title "Add HW2 description" --body "Adds hw2/README.md and updates the root README."
gh pr merge --merge --delete-branch   # merge the pull request into main
git switch main && git pull          # sync local main with the merged result
```

## Why a pull request

- `main` stays stable: changes arrive only after review.
- A PR shows the full diff and discussion in one place.
- Merging keeps the branch history visible in `git log --graph`.

## Result

The history of `main` contains the merge commit of the `feature/hw2` pull request:

```bash
git log --oneline --graph
```

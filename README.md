# The Masters' [Grasp-and-Lift EEG Detection](https://www.kaggle.com/c/grasp-and-lift-eeg-detection)

## Git Rules

### Branching and merging

* Make a new branch for any goal or thread of development
    * `git checkout master`
    * `git checkout -b new-branch-name`
* Name the branch a short, dash-separated phrase that summarizes the branch's purpose
    * `graph-raw-data`
    * `update-contributors`
    * `fix-eeg-parsing`
* When the branch is complete, submit a pull request (PR) through Bitbucket
    * Designate some developer(s) to be PR reviewer(s)
    * When all reviewers approve and all issues are resolved, merge the branch and close the PR

### Commit messages

* One-line, under-50-character summary
    * Imperative tense: `Add EEG graphing script`, rather than `Adds EEG graphing script` or `New EEG graphing script`
    * One concept/feature/fix per commit (don't join two ideas like `Add EEG graphing script and fix EEG parsing`)
* Blank line under one-line summary
* Detailed explanation of changes
    * Sentences/paragraphs separated by blank lines
    * Imperative tense where possible
    * Wrap lines manually to keep them under 80 characters

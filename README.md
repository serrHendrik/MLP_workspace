# Upcoming deadline:
May, 13th
# TODO
* Task 2
* Task 3
* Report!

# Overleaf
#### Task 1
    https://www.overleaf.com/6877141131cfbwfvdwrskz
#### Task 2/3
    https://www.overleaf.com/5774999625pbqgxdpvybpq
# Git Basics
## Clone repo
Navigate to directory where you want to clone this repository and use the following command:

    git clone https://github.com/serrHendrik/MLP_workspace.git

## General workflow
#### Add changes locally
    git add .
#### Commit changes
    git commit -m "\<message\>"
#### Check if local repo is up to date + handle merge conflicts
    git pull origin master
#### push changes
    git push origin master


## Extended workflow with personal local branch
Working on a personal local branch is recommended. It makes it easier to rollback or simply discard wrong commits by deleting the local branch.
#### Create local branch
    git pull origin master (pull updates before starting to work)
    git checkout -b my_local_branch
#### Do work on local branch
    After doing your work, add and commit.
#### Check for updates on local master
    git checkout master
    git pull origin master
#### merge master with local personal branch
    git merge master my_local_branch
#### push changes
    git push origin master
#### delete local branch
    git branch -d my_local_branch
#### to check the branch you're working on
    git branch

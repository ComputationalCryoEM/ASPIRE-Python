
# Coding Guidelines

This document contains the coding guidelines for the ASPIRE Python project.

## Coding Style

In order to keep the code consistent, please read and follow [PEP 8 Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008)
and [PEP 257 Docstring Conventions](https://www.python.org/dev/peps/pep-0257/) guidelines. [This webpage](https://realpython.com/documenting-python-code/) is a good guide for documenting Python code.
For an editor with some automatic checking of style problems, [Pycharm from Jet Brains](https://www.jetbrains.com/pycharm/)
is a good candidate which has the professional version for academic users with free license fee.

In the future components of PEP8 and PEP257 will be checked programmatically.  Documentation of a standard tool to perform
these operations locally will be provided in Google Doc to developers at that time.

## Good Practices Reading

Some thoughts on Scientific Computing are discussed in Greg Wilson and collaborators' paper as below:

 1. [Good Enough Practices in Scientific Computing]( https://doi.org/10.1371/journal.pcbi.1005510)
 2. [Best Practices for Scientific Computing]( https://doi.org/10.1371/journal.pbio.1001745)

## Source Code Control

As an open source software, we use Git with GitHub for our source code control.
The following links are useful tutorials.
1. Git tutorials from [Atlassian](https://www.atlassian.com/git/tutorials) and [Git-SCM](https://git-scm.com/docs/gittutorial)
2. GitHub guides from [GitHub](https://guides.github.com/)

## Git Workflow

Our project follows a reduced [Gitflow Workflow](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow).
Please submit any Pull Requests (PR) against the `develop` branch.

![Gitflow Diagram](https://wac-cdn.atlassian.com/dam/jcr:61ccc620-5249-4338-be66-94d563f2843c/05%20(2).svg?cdnVersion=357)

The basic idea is that we have `feature` branches,  `develop` as an integration focused branch, and `master` for releases.
We do not currently have a need for a dedicated release or staging branch; in our case `develop` also serves this purpose.
Generally feature branches should branch off of the latest `develop` branch.

Developers are welcome to maintain their own fork, but this is not required.

When `develop` is at a milestone where we want to cut a release,
following the controls below we will commit with a
[PEP440](https://www.python.org/dev/peps/pep-0440/)
compliant `Major.Minor` version scheme
and complete a reviewed merge to `master`.
Once in `master` a formal Git (version) tag can be applied to the commit,
and at this point the release would be a candidate to upload to PyPI.

The process for these steps will be documented in a Google Doc.

###  Git Workflow Controls

To facilitate effective integration, merges into `develop` must occur as a pull request that is approved by two reviewers.
One reviewer can be anyone, but should probably be a developer related to the effort if that is applicable.
The second reviewer is a designated codeowner described in the `CODEOWNERS` file.
More information on codeowners can be [found on GitHub.](https://help.github.com/en/github/creating-cloning-and-archiving-repositories/about-code-owners)

Additionally, all outstanding requested changes and comments should be resolved
by the requesting reviewer before merging.
Pushing new changes should trigger a re-review process
of changed files to prevent oversights.  

Code should be passing integration and unit tests as defined in code on all
supported platforms by the evolving Continuous Integration (CI) systems,
or be explicitly exempted by a codeowner.
This is both for code quality and so CI maintenance is not overlooked.

All of these controls will be managed automatically by the GitHub server.

Releases and merges to `master` should be coordinated between developers and the project PI.
This will be documented with an additional required review from the PI when merging into `master`.

### Git Workflow Courtesy

Contributors are responsible for keeping their working repositories up to date with the main GitHub repo.
This is particularly important during active development,
as it will reduce the likelihood and extent of code conflicts
when new features are implemented.

If a piece of work is known to change files that will effect other developers' active work, they should be invited to the review or tagged,
so that the most effective order of integration can be established.  In the case of your own work having integration
concerns, again, inviting other developers to discuss might be helpful.

In the case a codeowner will not be available, such as (sickeness, vacation, travel, and so on)
and an emergency patch can not be delayed, a simple explicit consensus among
the remaining developers should suffice.

Do not push or merge over another working branch unless explicitly permitted to do so by the authors.
Instead, when working on a shared branch submit a Pull Request to the other party.

## Project Collaboration

We will for a time attempt using
[GitHub project boards](https://help.github.com/en/github/managing-your-work-on-github/about-project-boards)
in conjunction with
[Issues](https://help.github.com/en/github/managing-your-work-on-github/about-issues)
to document tasks in various stages.
Generally this will look like a board collecting related tasks representing a conceptual project or
actual milestone that comes out of our meetings.  For example, an event, release, or large feature set
with many components all are reasonable project boards.  This is for planning purposes and to facilitate coordination
in the remote environment. If it helps we can keep doing it so long as the time and effort is accounted for.


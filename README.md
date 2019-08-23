# BugBuddy
BugBuddy is an experimental Deep Learning Fault Localization tool that automatically blames code for a test failure.

### Problem
Developers spend 30-50% of their time debugging [<sup>[1]</sup>](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.444.9094&rep=rep1&type=pdf).  In large repositories, it's possible to break tests that you didn't even know exist.  BugBuddy uses deep learning to automatically blame the code that caused the test failure.  This expedites and localizes your debugging journey for improved developer velocity. 

### Quick guide

 - Initialize your repository with `bugbuddy initialize /path/to/source -i "env-activation-command" -t "test-runner-command" -s "project_name"`
 - Generate synthetic training data with `bugbuddy generate /path/to/source`
 - Once that is done, you can have BugBuddy automatically blame code for each test failure as soon as they happen with `bugbuddy watch /path/to/source`.

### Features
- An AST manipulation logic to generate synthetic edits on your repository.
- Automatic test-runner whenever a source file is altered using [watchdog](https://github.com/gorakhargosh/watchdog)
- Comprehensive data collection on a per-commit basis.

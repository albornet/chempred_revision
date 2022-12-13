import os

gitignore_content = \
"""# Ignore everything in this directory
*
# Except this file
!.gitignore
"""

dirs_to_add_to_git = ['./open-nmt/config', './data']

for top_dir in dirs_to_add_to_git:
    for folder, subfolders, files in os.walk(top_dir):
        if len(subfolders) == 0:
            with open(os.path.join(folder, '.gitignore'), 'w') as f:
                f.write(gitignore_content)
            
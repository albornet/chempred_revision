import os

gitignore_content = \
"""# Ignore everything in this directory
*
# Except this file
!.gitignore
"""

for folder, subfolders, files in os.walk('./open-nmt/config'):
    if len(subfolders) == 0:
        with open(os.path.join(folder, '.gitignore'), 'w') as f:
            f.write(gitignore_content)

for folder, subfolders, files in os.walk('./open-nmt/config'):
    if len(subfolders) == 0 and 'original' not in folder:
        with open(os.path.join(folder, '.gitignore'), 'w') as f:
            f.write(gitignore_content)



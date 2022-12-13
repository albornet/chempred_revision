import os

gitignore_content = \
"""# Ignore everything in this directory
*
# Except this file
!.gitignore
"""

config_dir = './open-nmt/config'
data_dir = './data'

for top_dir in ['config_dir', 'data_dir']:
    for folder, subfolders, files in os.walk(config_dir):
        if len(subfolders) == 0:
            with open(os.path.join(folder, '.gitignore'), 'w') as f:
                f.write(gitignore_content)
            
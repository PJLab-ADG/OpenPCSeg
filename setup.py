import os
import subprocess
from setuptools import setup


def get_git_commit_number():
    if not os.path.exists('.git'):
        return '0000000'

    cmd_out = subprocess.run(['git', 'rev-parse', 'HEAD'], stdout=subprocess.PIPE)
    git_commit_number = cmd_out.stdout.decode('utf-8')[:7]
    return git_commit_number

def write_version_to_file(version, target_file):
    with open(target_file, 'w') as f:
        print('__version__ = "%s"' % version, file=f)

if __name__ == '__main__':
    version = '1.0.0+%s' % get_git_commit_number()
    write_version_to_file(version, '__version__')

    setup(
        name='pcseg',
        version=version,
        description='PCSeg: Open Source Point Cloud Segmentation Toolbox and Benchmark',
        install_requires=[
            'tqdm',
        ],
        author='PJLAB-ADG',
        license='Apache License',
        packages=['pcseg']
    )

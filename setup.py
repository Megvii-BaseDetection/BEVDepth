from setuptools import find_packages, setup

with open('README.md', 'r') as fh:
    long_description = fh.read()

setup(name='BEVDepth',
      version='0.0.1',
      author='Megvii',
      author_email='liyinhao@megvii.com',
      description='Code for BEVDepth',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url=None,
      packages=find_packages(),
      classifiers=[
          'Programming Language :: Python :: 3',
          'Operating System :: OS Independent',
      ],
      install_requires=[])

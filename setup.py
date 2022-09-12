from setuptools import setup

setup(
  name='noisy_or',
  version='0.0.1',
  packages=['noisy_or'],
  python_requires='>=3.8',
  url='',
  package_dir={'noisy_or': 'noisy_or'},
  license='GPL-3.0',
  author='Saurabh Mathur',
  author_email='saurabhmathur96@gmail.com',
  description='',
  install_requires=open('requirements.txt', 'r').read().splitlines()
)

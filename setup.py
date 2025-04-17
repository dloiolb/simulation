from setuptools import setup, find_packages
from pathlib import Path

def parse_requirements():
    requirements = Path('requirements.txt').read_text().splitlines()
    return requirements

setup(
    name='simulation',
    version='0.1',
    packages=find_packages(),
    install_requires=parse_requirements(),
)

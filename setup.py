from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    with open(file_path ,  encoding="utf-8") as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.strip() for req in requirements if req.strip() and not req.startswith("#")]
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    return requirements


setup(
    name='mlProject',
    version='0.0.1',
    author='Chirayu Kapoor',
    author_email='chirayukapoor2002@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)

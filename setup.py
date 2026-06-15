from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).resolve().parent


def read_requirements():
    requirements_path = ROOT / "requirements.txt"
    lines = requirements_path.read_text().splitlines()
    requirements = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        requirements.append(stripped)
    return requirements


setup(
    name="robotcollab",
    version="1.0",
    packages=find_packages(),
    description="Code for Paper: RoCo: Multi-Robot Collaboration with Large Language Models",
    url="https://github.com/MandiZhao/robot-collab.git",
    author="Mandi Zhao",
    install_requires=read_requirements(),
    extras_require={
        "visualization": ["open3d"],
        "metaworld": [
            'gymnasium>=1.1,<2; python_version >= "3.10"',
            'metaworld==3.0.0; python_version >= "3.10"',
        ],
    },
)

from setuptools import setup, find_packages

setup(
    name="indoorplants",
    version="1.4",
    description="Tools for data analysis and model validation",
    author="Abe Lerman",
    url="https://github.com/lermana/indoorplants",
    packages=find_packages(),
    install_requires=["numpy",
                      "scipy",
                      "pandas",
                      "scikit-learn",
                      "matplotlib",
                      "requests"]
    )

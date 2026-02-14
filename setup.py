from setuptools import find_packages, setup

setup(
    name="vbn",
    version="0.3.0",
    description="Vectorized Bayesian Network",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Giovannibriglia/VectorizedBayesianNetwork",
    license="MIT",
    author="Giovanni Briglia",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0",
    ],
    include_package_data=True,
    package_data={"vbn": ["configs/**/*.yaml"]},
)

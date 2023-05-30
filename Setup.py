import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="dygpy",
    version="1.0.0",
    author="FardinBahreini",
    author_email="fardinbhi@gmail.com",
    description="A library for creating and calculating graphs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fardinbh/dygpy",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

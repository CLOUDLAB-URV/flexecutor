from setuptools import setup, find_packages

setup(
    name="flexecutor",
    version="0.1.0",
    author="Daniel Barcelona, Ayman Bourramouss, Enrique Molina, Stepan Klymonchuk (DAGium donation)",
    author_email="daniel.barcelona@urv.cat, ayman.bourramouss@urv.cat, enrique.molina@urv.cat",
    description="A flexible and DAG-optimized executor over Lithops",
    url="https://github.com/abourramouss/smart-datastaging-layer/",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    license="Apache 2.0",
    license_files=["LICENSE"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering",
        "Topic :: System :: Distributed Computing",
    ],
    python_requires=">=3.6",
    install_requires=[
        "lithops",
        "cloudpickle",
        "deap",
        "scipy",
        "numpy",
        "matplotlib",
        "networkx",
        "pandas"
    ],
    extras_require={
        "examples": [
            "S3path",
            "boto3"
        ]
    },
    packages=find_packages(),
    package_dir={"": "."},
)

from setuptools import setup, find_packages

setup(
    name="canya",
    version="0.0.2",
    description="CANYA a hybrid neural-network to predict nucleation propensities",
    author='Mike Thompson',
    author_email='mjthompson at ucla dot edu',
    packages=find_packages(),
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3 :: Only",
    ],
    # Do not install tensorflow here, because might want to use tensorflow or
    # tensorflow-cpu.
    package_data={"canya" : ["models/*"]},
    install_requires=[
        "numpy==1.19.5",
        "pandas<=1.3.3",
        "tensorflow==2.6.0",
        "keras==2.6.0",
        "protobuf<=3.20"],
    python_requires=">=3.6, <4",
    entry_points={"console_scripts": ['canya=canya.__main__:main']}
)
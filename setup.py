import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ml_downscaling_emulator",
    version="0.0.1",
    author="Henry Addison",
    author_email="henry.addison@bristol.ac.uk",
    description="A package for downscaling precipitation forecasts",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/henryaddison/ml-downscaling-emulation",
    project_urls={
        "Bug Tracker": "https://github.com/henryaddison/ml-downscaling-emulation/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    entry_points={
        'console_scripts': [
            'preprocess = ml_downscaling_emulator.bin.preprocess:cli',
            'evaluate = ml_downscaling_emulator.bin.evaluation:cli',
            'mlde = ml_downscaling_emulator.bin:app'
        ],
    },
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)

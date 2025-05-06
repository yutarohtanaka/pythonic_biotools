from setuptools import setup, find_packages

setup(
    name="xenofilterR",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "pysam",
        "pandas",
        "numpy"
    ],
    entry_points={
        "console_scripts": [
            "xenofilter=xenofilter.cli:main"
        ]
    },
    description="Filter mouse reads from human xenograft RNA-seq or DNA-seq BAMs",
    author="Yutaro Tanaka",
)

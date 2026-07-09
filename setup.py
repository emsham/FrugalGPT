from setuptools import setup, find_packages

provider_extras = {
    "ai21": ["ai21"],
    "anthropic": ["anthropic"],
    "cohere": ["cohere"],
    "google": ["google-generativeai"],
}

scoring_extras = ["accelerate", "evaluate", "torch", "transformers"]

setup(
    name='FrugalGPT',
    version='0.0.1',
    author='Lingjiao Chen, Matei Zaharia, and James Zou',
    author_email='lingjiao@stanford.edu',
    description='The FrugalGPT library',
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    data_files=[
        ("config", ["config/serviceinfo.json", "config/serviceidmap.json"]),
    ],
    install_requires=[
        'numpy',
        'smart-open',
        'scikit-learn',
        'scipy',
        'pandas',
        'requests',
        'sqlitedict',
        'tqdm',
    ],
    extras_require={
        **provider_extras,
        "scoring": scoring_extras,
        "all": sorted(set(scoring_extras + sum(provider_extras.values(), []))),
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)

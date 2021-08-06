import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="u2net_pytorch",
    version="0.0.1",
    author="Xuebin Qin, Serhii Korzh",
    author_email="serhii.korzh@aalto.fi",
    description="Python PyTorch implementation of the different U-2-Net models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/souserge/U-2-Net",
    project_urls={
        "Bug Tracker": "https://github.com/souserge/U-2-Net/issues",
    },
    classifiers=["Programming Language :: Python :: 3"],
    install_requires=[
        "numpy",
        "scikit-image",
        "opencv-contrib-python",
        "pillow",
        "torch",
        "torchvision",
        "gradio",
        "gdown",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.8",
)

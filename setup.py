from setuptools import find_packages, setup

# Settings
import template

setup(
    name='template',  # name of pypi package
    version=template.__version__,  # version of pypi package
    python_requires='>=3.7',
    license='MIT',
    description="Template codebase for various deep learning tasks.",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="deep learning",
    author='Zili Wang',
    author_email='1348131731@qq.com',
    packages=find_packages(exclude=("configs",
                                    "dataset",
                                    "evaluation",
                                    "modelings",
                                    "optim",
                                    "result",
                                    "runs",
                                    "wandb",
                                    "template.egg-info")),  # required
    install_requires=["rich", "fire", "accelerate", "torch>=1.10.0"],
    include_package_data=True,
    entry_points={'console_scripts': ["template=template.template_cli:main",
        'show_gpu=template.launcher:show_gpu']})

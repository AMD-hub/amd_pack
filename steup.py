from setuptools import setup, find_packages

setup(
    name='amd_pack',
    version='0.1',
    description='A Python package for aleatory and related utilities',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/AMD-hub/amd_pack',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.18.0',         # Include numpy
        'scipy>=1.4.0',          # Include scipy
        'pandas>=1.0.0',         # Include pandas
        'matplotlib>=3.1.0',     # Include matplotlib
    ],
    classifiers=[
        'Development Status :: 4 - Beta',  # Choose the appropriate status
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',  # Update if using a different license
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
    python_requires='>=3.7',  # Specify the Python versions you support
    include_package_data=True,  # Include non-code files specified in MANIFEST.in
    package_data={
        '': ['*.txt', '*.rst', '*.md'],
    },
)

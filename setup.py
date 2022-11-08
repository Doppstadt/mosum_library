from setuptools import find_packages, setup
setup(
    name='mosum library',
    packages=find_packages((include=['mosum_library'])),
    version='0.1.0',
    description='Mosum Library',
    author='Me',
    license='MIT',
    install_requires=[],
    setup_requires=['pytest-runner'],
    tests_require=['pytest==4.4.1'],
    test_suite='tests',
)
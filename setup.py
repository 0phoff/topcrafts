import os
import setuptools as setup
from pkg_resources import get_distribution, DistributionNotFound


def get_dist(pkgname):
    try:
        return get_distribution(pkgname)
    except DistributionNotFound:
        return None

def find_packages():
    return ['top'] + ['top.'+p for p in setup.find_packages('top')]

def find_scripts():
    return setup.findall('scripts')

def create_version():
    cwd = os.getcwd()

    with open(os.path.join(cwd, 'VERSION'), 'r') as f:
        version = f.read().splitlines()[0]
    with open(os.path.join(cwd, 'top', 'version.py'), 'w') as f:
        f.write(f'__version__ = \'{version}\'')

    return version


requirements = [
    'numpy',
    'scipy',
    'matplotlib',
    'tqdm',
]
pillow_req = 'pillow-simd' if get_dist('pillow-simd') is not None else 'pillow'
requirements.append(pillow_req)

version = create_version()
setup.setup(
    name='top',
    version=version,
    author='0phoff',
    description='TOPCraft: Bits and bops that I often need and dont want to rewrite',
    long_description=open('README.md').read(),
    packages=find_packages(),
    scripts=find_scripts(),
    install_requires=requirements,
)

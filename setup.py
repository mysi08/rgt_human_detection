from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'obt'

# Helper function to automatically crawl and include folder contents
def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append((os.path.join('share', package_name, path), [os.path.join(path, filename)]))
    return paths

extra_files = package_files('model')

setup(
    name=package_name,
    version='0.0.0',
    # FIX 1: Explicitly name the package package_name ('obt') instead of relying on find_packages()
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ] + extra_files,
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='rgt',
    maintainer_email='rgt@todo.todo',
    description='OBT object detection package',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'main_node = obt.main:main',
            'hdt_node = obt.main:main',
        ],
    },
)


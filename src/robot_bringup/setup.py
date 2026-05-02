from setuptools import setup, find_packages
import os
from glob import glob

package_name = "robot_bringup"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages",
            ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"),
            glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "urdf"),
            glob("urdf/*.urdf") + glob("urdf/*.xacro")),
        (os.path.join("share", package_name, "config"),
            glob("config/*.yaml")),
        (os.path.join("share", package_name, "rviz"),
            glob("rviz/*.rviz")),
        (os.path.join("share", package_name, "isaac"),
            glob("isaac/*.py")),
        (os.path.join("lib", package_name),
            glob("scripts/*.sh")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    entry_points={
        "console_scripts": [
        ],
    },
)

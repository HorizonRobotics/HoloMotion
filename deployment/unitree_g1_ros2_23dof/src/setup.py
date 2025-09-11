from setuptools import setup, find_packages
import os


package_name = "humanoid_control"

data_files = [
    ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
    ("share/" + package_name, ["package.xml"]),
]
# Add files from config, launch and model directories
for dir_name in ["config", "launch", "models"]:
    if os.path.exists(dir_name):  # Only process if directory exists
        for root, dirs, files in os.walk(dir_name):
            install_dir = os.path.join("share", package_name, root)
            list_entry = (install_dir, [os.path.join(root, f) for f in files])
            data_files.append(list_entry)

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(),
    data_files=data_files,
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Horizon Robotics",
    maintainer_email="maiyue01.chen@horizon.auto",
    description="Humanoid locomotion control package from Horizon Robotics",
    license="Apache License 2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            
            "policy_node_performance = humanoid_policy.policy_node_performance:main",
        ],
    },
)

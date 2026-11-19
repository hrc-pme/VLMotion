from setuptools import find_packages, setup


package_name = "vlpoint"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml", "README.md"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="PME",
    maintainer_email="pme@example.com",
    description="Stretch-side VLPoint conversation and image client",
    license="Apache-2.0",
)

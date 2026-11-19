from setuptools import find_packages, setup


package_name = "vlmotion_stt"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="HRC PME",
    maintainer_email="hrclab.nthu@gmail.com",
    description="Offline faster-whisper adapter for ReSpeaker utterances",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "speech_to_text = vlmotion_stt.speech_to_text:main",
        ],
    },
)

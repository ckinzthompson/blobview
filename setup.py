import setuptools
from setuptools.command.install import install as _install

with open("README.md") as f:
	ld = f.read()

#pip version
setuptools.setup(
	name="blobview",
	version="0.0.1",
	author="Colin Kinz-Thompson",
	author_email="colin.kinzthompson@rutgers.edu",
	description="Minimal biomolecular structure and density visualization",
	long_description=ld,
	long_description_content_type="text/markdown",
	license="GPLv3",
	package_data={"": ["LICENSE",]},
	url="",
	packages=['blobview'],
	python_requires='>=3.9',
	install_requires=["numba>=0.55","numpy>=1.22","vispy>=0.10","matplotlib>=3.5","PyQt5>=5.15"],
	classifiers=[
		"Programming Language :: Python :: 3",
		"License :: OSI Approved :: License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
		"Topic :: Scientific/Engineering :: Chemistry",
		"Topic :: Scientific/Engineering :: Biology",
	],
	include_package_data=True,
	zip_safe=True,
)

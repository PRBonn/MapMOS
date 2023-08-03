install:
	@pip install -v .

install-all:
	@pip install -v ".[all]"

uninstall:
	@pip -v uninstall mapmos

editable:
	@pip install scikit-build-core pyproject_metadata pathspec pybind11 ninja cmake
	@pip install --no-build-isolation -ve .

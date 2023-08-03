install:
	@pip install -v .

install-all:
	@pip install -v ".[all]"

uninstall:
	@pip -v uninstall mapmos

build-system:
	@pip install scikit-build-core pyproject_metadata pathspec pybind11 ninja cmake

editable: build-system
	@pip install --no-build-isolation -ve .

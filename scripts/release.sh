rm -rf dist/*
python3 setup.py sdist bdist_wheel
delocate-wheel -w fixed_wheels -v dist/*.whl

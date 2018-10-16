.DEFAULT_GOAL := list

list:
	@echo "Available options (you can use auto-complete):"
	@$(MAKE) -pRrq -f $(lastword $(MAKEFILE_LIST)) : 2>/dev/null | awk -v RS= -F: '/^# File/,/^# Finished Make data base/ {if ($$1 !~ "^[#.]") {print $$1}}' | sort | egrep -v -e '^[^[:alnum:]]' -e '^$@$$' | xargs

data:
	python ./utils/download_binaries.py; tree binaries

finufftpy:
	bash ./utils/install_finufftpy.sh

build-docs:
	sphinx-build -b html -E -a docs/source/ docs/build/

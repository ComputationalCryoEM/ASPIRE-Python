.DEFAULT_GOAL := list

list:
	@echo Available options:
	@$(MAKE) -pRrq -f $(lastword $(MAKEFILE_LIST)) : 2>/dev/null | awk -v RS= -F: '/^# File/,/^# Finished Make data base/ {if ($$1 !~ "^[#.]") {print $$1}}' | sort | egrep -v -e '^[^[:alnum:]]' -e '^$@$$' | xargs

data:
	python ./download_binaries.py

install:
	bash ./install.sh

build-docs:
	bash ./build_docs.sh


default:
	@echo; grep -I '#' makefile | grep -v makefile; echo

runtests: # Run the tests!
	py.test -v tests/

setupenv: # Set up the python3 virtualenv for the first time
	virtualenv -p python3 env

requires: # Install all the requirements
	pip -r requirements.txt


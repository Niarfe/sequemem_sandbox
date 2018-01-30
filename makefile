
default:
	@echo; grep -I '#' makefile | grep -v makefile; echo

runtests: # Run the tests!
	py.test -v tests/

setupenv: # Set up the python3 virtualenv for the first time
	virtualenv -p python3 env

requires: # Install all the requirements
	pip install -r requirements.txt


startenv: # Start the env virtual environment
	. env/bin/activate

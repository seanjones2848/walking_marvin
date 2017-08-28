FROM debian:latest

RUN apt-get update &&\
	apt-get install -y build-essential python-dev swig python-pygame python-pip git &&\
	pip install gym

CMD cd home &&\
	git clone https://github.com/pybox2d/pybox2d &&\
	cd pybox2d &&\
	python setup.py build &&\
	python setup.py install

ADD envs /usr/local/lib/python2.7/dist-packages/gym/envs

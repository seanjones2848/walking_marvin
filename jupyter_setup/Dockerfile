FROM jupyter/minimal-notebook

RUN apt-get install python-opengl

RUN conda install python=3.5 &&\
	conda install -c https://conda.anaconda.org/kne pybox2d

RUN pip install gym matplotlib pandas pygame scipy pyglet

ADD envs /opt/conda/lib/python3.5/site-packages/gym/envs

CMD start-notebook.sh

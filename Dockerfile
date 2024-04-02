# $DEL_BEGIN

# ####### 👇 SIMPLE SOLUTION (x86 and M1) 👇 ########
# FROM python:3.8.12-buster

# WORKDIR /prod

# COPY requirements_old.txt requirements_old.txt
# RUN pip install -r requirements_old.txt

# COPY taxifare taxifare
# COPY setup.py setup.py
# RUN pip install .

# COPY Makefile Makefile
# RUN make reset_local_files

# CMD uvicorn taxifare.api.fast:app --host 0.0.0.0 --port $PORT

####### 👇 OPTIMIZED SOLUTION (x86)👇 #######

# tensorflow base-images are optimized: lighter than python-buster + pip install tensorflow
FROM tensorflow/tensorflow:latest
# OR for apple silicon, use this base image instead
#FROM armswdev/tensorflow-arm-neoverse:latest


WORKDIR /prod

ENV PORT=8000

# We strip the requirements from useless packages like `ipykernel`, `matplotlib` etc...
#COPY requirements_old.txt requirements_old.txt
#RUN pip install -r requirements_old.txt
#USER dockuser
COPY aitrading aitrading
COPY setup.py setup.py
COPY aitrading/models aitrading/models


#COPY peotry.lock poetry.lock
COPY pyproject.toml pyproject.toml


RUN pip install poetry
RUN poetry install
#RUN pip install uvicorn

COPY Makefile Makefile
#RUN make reset_local_files

CMD uvicorn aitrading.api.main:app --host 0.0.0.0 --port $PORT
#CMD make run_fast_api
# $DEL_END

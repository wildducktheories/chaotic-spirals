FROM python:3.7
RUN mkdir /home/spirals
WORKDIR /home/spirals
COPY requirements.txt .
RUN pip3 install -r requirements.txt
CMD jupyter notebook --allow-root --ip=0.0.0.0
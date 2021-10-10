FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8

RUN pip3 install --upgrade pip
RUN echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | tee /etc/apt/sources.list.d/tensorflow-serving.list && \
curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | apt-key add -
RUN apt-get update && apt-get install tensorflow-model-server


WORKDIR /raster_dtfr
COPY . /raster_dtfr
RUN pip3 --no-cache-dir install -r requirements.txt
ENV MODEL_NAME=raster_dtfr
ENV MODEL_DIR=/raster_dtfr/serving_data
RUN ["chmod", "+x", "/raster_dtfr/start.sh"]
CMD /raster_dtfr/start.sh

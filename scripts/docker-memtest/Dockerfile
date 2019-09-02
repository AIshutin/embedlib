FROM python:3.7

RUN pip3 install torch torchvision #torch==1.2.0+cpu torchvision==0.4.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

RUN pip3 install numpy

COPY embedlib /app/embedlib/embedlib
COPY setup.py /app/embedlib/
RUN pip3 install /app/embedlib/

RUN pip3 install laserembeddings
RUN python3 -m laserembeddings download-models

#COPY ru-6-attentions /app/model/
COPY laser-3lays-twitt-mrr0.67 /app/model/

COPY memory_checker.py /app/memory_checker.py

CMD ["python3", "/app/memory_checker.py"]

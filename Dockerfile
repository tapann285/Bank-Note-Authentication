FROM continuumio/anaconda3:4.4.0
RUN pip install -r requirements.txt
CMD python flask_api.py
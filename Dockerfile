FROM python:3.10

ENV APP_HOME /app
WORKDIR $APP_HOME
ENV PYTHONPATH /

# Get necessary system packages
RUN apt-get update \
  && apt-get install --no-install-recommends --yes \
     build-essential \
     python3 \
     python3-pip \
     python3-dev \
     mariadb-client \
  && rm -rf /var/lib/apt/lists/*


RUN pip3 install --upgrade mysql-connector-python
RUN python3 -m pip install --upgrade pip

COPY requirements.txt .
RUN pip3 install --compile --no-cache-dir -r requirements.txt

# Copy over code
COPY sql.sh /
COPY baseball.sql .

RUN chmod +x /sql.sh

ENTRYPOINT ["/sql.sh"]

COPY final.py /


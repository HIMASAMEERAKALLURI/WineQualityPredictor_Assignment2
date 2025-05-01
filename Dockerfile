# Use an official OpenJDK base image
FROM openjdk:11

# Install Spark dependencies
ENV SPARK_VERSION=3.4.1 \
    HADOOP_VERSION=3

RUN apt-get update && apt-get install -y curl wget && \
    curl -L https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz \
    | tar -xz -C /opt/ && \
    mv /opt/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} /opt/spark

ENV SPARK_HOME=/opt/spark
ENV PATH=$SPARK_HOME/bin:$PATH

# Copy your jar or source code into the container
WORKDIR /app
COPY . /app

# Default command to run (you can override it during spark-submit)
CMD ["bash"]

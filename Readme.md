# Installation Steps

# Install Java (if not installed)

# Check Java installation:

java -version

# If not installed, install Java 11:

sudo apt update
sudo apt install openjdk-11-jdk

**Install Spark Locally**

Download Spark 3.4.1 (pre-built for Hadoop 3):

wget https://archive.apache.org/dist/spark/spark-3.4.1/spark-3.4.1-bin-hadoop3.tgz

**Extract the Spark archive**

tar -xvzf spark-3.4.1-bin-hadoop3.tgz

**Move Spark to /opt (optional)**

sudo mv spark-3.4.1-bin-hadoop3 /opt/spark

**Set Spark environment variables**

echo 'export SPARK_HOME=/opt/spark' >> ~/.bashrc
echo 'export PATH=$SPARK_HOME/bin:$PATH' >> ~/.bashrc
source ~/.bashrc

**Clone the Project**

git clone https://github.com/HIMASAMEERAKALLURI/WineQualityPredictor.git
cd Wine-Quality-prediction

**Build the Project (Optional if Already Built)**


**Compile your Java files**

javac -classpath "$SPARK_HOME/jars/*" -d target/ src/TrainModel.java

**Create a JAR**

jar cvf target/wine-quality-predictor-1.0.jar -C target/ .


**################**
 Running the Project
**################**
After compiling:

***Run the Spark job locally using spark-submit**

spark-submit \
  --class TrainModel \
  --master local[4] \
  target/wine-quality-predictor-1.0.jar



--master local[4] : Run locally with 4 CPU cores.





Make sure Spark binaries are properly extracted and added to PATH.

If Spark commands are not recognized, check your environment variables.

Ensure your dataset file path in the code matches where your CSV is stored.
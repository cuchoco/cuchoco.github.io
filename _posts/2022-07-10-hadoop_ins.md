---
title:  "Hadoop3 싱글노드 & Pseudo-Distributed 설치"
excerpt: "Apache Hadoop"
categories:
  - Data engineering

toc: true
toc_sticky: true
---

# Hadoop3 설치방법
ubuntu 20.04 LTS 사용.   
Hadoop-3.2.3 버젼을 설치했으며  
<https://hadoop.apache.org/docs/r3.2.3/hadoop-project-dist/hadoop-common/SingleCluster.html>
를 참고했다.

## 1. JAVA install
``` bash
sudo apt install openjdk-8-jdk
sudo vi /etc/profile
export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
source /etc/profile
```

## 2. Hadoop Download
```bash
wget https://www.apache.org/dyn/closer.cgi/hadoop/common/hadoop-3.2.3/hadoop-3.2.3.tar.gz
tar xzvf hadoop-3.2.3.tar.gz
mv hadoop-3.2.3 /home/cuchoco/hadoop

cd ~/hadoop
vi etc/hadoop/hadoop-env.sh
export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
source etc/hadoop/hadoop-env.sh

# 실행해보기
bin/hadoop  
```

## 3. Pseudo-Distributed Operation
다음과 같은 configuration을 추가해준다 

```bash
vi etc/hadoop/core-site.xml
```
```
<configuration>
    <property>
        <name>fs.defaultFS</name>
        <value>hdfs://localhost:9000</value>
    </property>
</configuration>
```
```bash
vi etc/hadoop/hdfs-site.xml
```
```
<configuration>
    <property>
        <name>dfs.replication</name>
        <value>1</value>
    </property>
</configuration>
```

### ssh 설정

```bash
ssh-keygen -t rsa -P '' -f .ssh/id_rsa
cat .ssh/id_rsa.pub >> .ssh/authorized_keys
chmod 0600 .ssh/authorized_keys

# 비밀번호 없이 접속 되는지 확인
ssh localhost
```

### HDFS 서비스 접속
```bash
bin/hdfs namenode -format
sbin/start-dfs.sh
# http://localhost:9870/ 접속해보기

# 접속이 잘 된다면
bin/hdfs dfs -mkdir /user
bin/hdfs dfs -mkdir /user/<username>
```

## 4. YARN set up
```bash
vi etc/hadoop/mapred-site.xml
```
```
<configuration>
    <property>
        <name>mapreduce.framework.name</name>
        <value>yarn</value>
    </property>
    <property>
        <name>mapreduce.application.classpath</name>
        <value>$HADOOP_MAPRED_HOME/share/hadoop/mapreduce/*:$HADOOP_MAPRED_HOME/share/hadoop/mapreduce/lib/*</value>
    </property>
</configuration>
```
```bash
vi etc/hadoop/yarn-site.xml
```
```
<configuration>
    <property>
        <name>yarn.nodemanager.aux-services</name>
        <value>mapreduce_shuffle</value>
    </property>
    <property>
        <name>yarn.nodemanager.env-whitelist</name>
        <value>JAVA_HOME,HADOOP_COMMON_HOME,HADOOP_HDFS_HOME,HADOOP_CONF_DIR,CLASSPATH_PREPEND_DISTCACHE,HADOOP_YARN_HOME,HADOOP_HOME,PATH,LANG,TZ,HADOOP_MAPRED_HOME</value>
    </property>
</configuration>
```
모두 추가하고 난 뒤에

```bash
sbin/start-yarn.sh
# http://localhost:8088/ 접속 확인
```
HDFS 서비스 포트 9870   
yarn 포트 8088

내 로컬 vscode에 port를 뚫어서 연결 해봤다.

<p align="center"><img src="/assets/images/hadoop/hadoop_ins1.png" height=150></p>

HDFS 서비스 접속

<p align="center"><img src="/assets/images/hadoop/hadoop_ins2.png" height=300></p>


YARN 접속

<p align="center"><img src="/assets/images/hadoop/hadoop_ins3.png" height=300></p>
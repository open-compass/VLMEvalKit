FROM registry.h.pjlab.org.cn/ailab-evalservice/vlmevalkit:xj-dev-v0.0.1

ADD . /app/vlmevalkit
WORKDIR /app/vlmevalkit

RUN pip install -r requirements.txt -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
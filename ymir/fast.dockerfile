FROM youdaoyzbx/ymir-executor:ymir2.1.0-mmrotate-cu113-base

COPY . /app
RUN cd /app && \
    pip install --no-cache-dir -r requirements.txt && \
    mkdir -p /img-man && \
    mv ymir/img-man/*.yaml /img-man
# mkdir /weights && \
# mv ymir/weights/*.pth /weights && \
# mkdir -p /root/.cache/torch/hub/checkpoints && \
# mv ymir/weights/imagenet/*.pth /root/.cache/torch/hub/checkpoints

ENV PYTHONPATH=.
WORKDIR /app

RUN echo "python3 ymir/start.py" > /usr/bin/start.sh
CMD bash /usr/bin/start.sh

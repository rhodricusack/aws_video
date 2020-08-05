FROM continuumio/anaconda3

SHELL [ "/bin/bash", "-c" ]

# Install SSH for remote debugging
RUN apt-get update && apt-get install -y openssh-server
RUN mkdir /var/run/sshd
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd
EXPOSE 22
RUN service ssh restart


# Private key login
RUN mkdir /root/.ssh
COPY authorized_keys /root/.ssh/authorized_keys

# Get ready for our program
RUN mkdir /app
WORKDIR /app

# CONDA package installation from environment.yml
COPY environment.yml /tmp/environment.yml
WORKDIR /tmp
RUN [ "conda", "env", "create" ]

# Pycharm helpers
COPY pycharm_helpers /root/.pycharm_helpers

# Pip install from requirements.txt
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN [ "/bin/bash", "-c", "source activate infant_rekognition && pip install -r requirements.txt" ]

# FFMPEG install from PPA
RUN mkdir /opt/ffmpeg
COPY ffmpeg-release-64bit-static.tar.xz /opt/ffmpeg
RUN tar -xvf /opt/ffmpeg/ffmpeg-release-64bit-static.tar.xz -C /opt/ffmpeg
RUN ln -s /opt/ffmpeg/ffmpeg-3.4-64bit-static/ffmpeg /usr/bin/ffmpeg
RUN ln -s /opt/ffmpeg/ffmpeg-3.4-64bit-static/ffprobe /usr/bin/ffprobe

# Code base
COPY . /app

RUN mkdir /root/.aws
COPY .setkeys /root/.aws/config

# Just wait for debug connection
CMD ["/usr/sbin/sshd", "-D"]
#CMD ["printenv"]
#CMD [ "/bin/bash", "-c", "source activate infant_rekognition && python infant_face_rekognition_nographics.py" ]

FROM python:3.10
LABEL author=tssujt
RUN apt update && apt install -y telnet vim
WORKDIR /app
COPY . .
RUN python -m pip install --no-cache-dir -r requirements.txt
CMD ["/bin/bash"]

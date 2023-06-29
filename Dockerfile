FROM ultralytics/ultralytics:latest-cpu

WORKDIR /code/app
 
ADD . .
 
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --no-cache-dir --upgrade -r requirements.txt

CMD ["python3", "web.py"]

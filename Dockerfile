FROM ultralytics/ultralytics:latest-cpu

WORKDIR /code/app
 
ADD . .

# -q 防止老版本docker build失败
RUN pip install -q -i https://pypi.tuna.tsinghua.edu.cn/simple --no-cache-dir --upgrade -r requirements.txt

CMD ["python3", "web.py"]

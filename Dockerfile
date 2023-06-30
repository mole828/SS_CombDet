FROM ultralytics/ultralytics:latest-cpu

WORKDIR /code/app
 
ADD . .

# 老版本docker:
#   build: 添加 -q 参数 防止pip安装使用多线程文字
#   run: 添加 --security-opt seccomp:unconfined 需要加入这个防止线程创建失败
RUN pip install -q -i https://pypi.tuna.tsinghua.edu.cn/simple --no-cache-dir --upgrade -r requirements.txt

CMD ["python3", "web.py"]

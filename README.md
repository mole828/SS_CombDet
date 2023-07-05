# SS_CombDet

## 部署命令

```sh
docker stop MLColorDetect1 && docker rm -v MLColorDetect1 && docker run -d -p 50001:8000 --restart always --cpus 4 -m 1g --security-opt seccomp:unconfined --name MLColorDetect1 ss_combdet:v1.1

docker stop MLColorDetect2 && docker rm -v MLColorDetect2 && docker run -d -p 50002:8000 --restart always --cpus 4 -m 1g --security-opt seccomp:unconfined --name MLColorDetect2 ss_combdet:v1.1

docker stop MLColorDetectDev && docker rm -v MLColorDetectDev && docker run -d -p 60001:8000 --restart always --cpus 1 -m 512m --name MLColorDetectDev ss_combdet:v1.1
```
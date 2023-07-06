class DockerCommandMaker:
    '''
    构造实际环境部署命令
    '''
    def __init__(self, container: str, image:str, ports:list[tuple[int,int]]) -> None:
        self.container = container
        self.image = image
        self.ports = ports

    def __str__(self) -> str:
        return " && ".join([
            f"docker stop {self.container}",
            f"docker rm -v {self.container}",
            ' '.join([
                f"docker run -d",
                *[f"-p {a}:{b}" for a,b in self.ports],
                "--restart always --cpus 4 -m 1g",
                "--security-opt seccomp:unconfined",
                f"--name {self.container} {self.image}",
            ])
        ])
    
if __name__ == '__main__':
    image = 'ss_combdet:v1.2'
    serve_port = 8000
    for container,port in [
        ('MLColorDetectDev', 60001),
        ('MLColorDetect1',   50001),
        ('MLColorDetect2',   50002),
    ]:
        print(DockerCommandMaker(container,image,[(port,serve_port)]))
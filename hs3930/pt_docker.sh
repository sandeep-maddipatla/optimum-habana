
#!/bin/bash

docker_version_default="1.20.0"
pt_version_default="2.5.1"
build_version_default="6"

if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
  echo "Usage: $0 [docker_version] [pt_version]"
  echo "  docker_version (optional): Docker version (default: $docker_version_default)"
  echo "  pt_version (optional): PyTorch version (default: $pt_version_default)"
  echo "  build_version (optional): Build version (default: $build_version_default)"
  exit 0
fi

docker_version=${1:-$docker_version_default}
pt_version=${2:-$pt_version_default}
build_version_default=$(curl -s https://artifactory-kfs.habana-labs.com:443/artifactory/docker-local/${docker_version}/ubuntu22.04/habanalabs/pytorch-installer-${pt_version}/ | grep -Po "${docker_version}-[0-9]*" | cut -d '-' -f 2 | sort -n | tail -1)
build_version=${3:-$build_version_default}

echo "  docker_version: Docker version : $docker_version"
echo "  pt_version: PyTorch version : $pt_version"
echo "  build_version: Build version :$build_version"

docker ps | grep -wq pytorch_gaudi2 2>/dev/null
if [ $? == 0 ]; then
  docker stop pytorch_gaudi2 2>/dev/null
  docker rm pytorch_gaudi2 2>/dev/null
  sleep 2
fi

script_dir=$(dirname "$0")
echo "artifactory-kfs.habana-labs.com/docker-local/${docker_version}/ubuntu22.04/habanalabs/pytorch-installer-${pt_version}:${docker_version}-${build_version}"
docker run -it --runtime=habana -e HABANA_VISIBLE_DEVICES="all" -e OMPI_MCA_btl_vader_single_copy_mechanism=none -e HTTP_PROXY=http://proxy-dmz.intel.com:911 \
   -e HTTPS_PROXY=http://proxy-dmz.intel.com:912 --rm --cap-add=sys_nice --net=host --ipc=host --privileged \
   -e no_proxy=localhost,0.0.0.0 \
   -v $script_dir:/root/eager_vs_lazy_loss \
   -v /data/:/data2/ -v /datasets/data:/data --name pytorch_gaudi2 --workdir=/root/ \
   artifactory-kfs.habana-labs.com/docker-local/${docker_version}/ubuntu22.04/habanalabs/pytorch-installer-${pt_version}:${docker_version}-${build_version}
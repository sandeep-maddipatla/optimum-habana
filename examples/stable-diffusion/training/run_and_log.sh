make_unique_dir() {
  base="$1"
  count=1
  dir="${base}_${count}"

  # Loop until a non-existing directory name is found
  while [ -d "$dir" ]; do
    dir="${base}_$count"
    ((count++))
  done

  mkdir -p "$dir"
  echo "$dir"
}

run() {
    tag=result
    result_dir=$(make_unique_dir ${tag})
    echo Using ${result_dir}
    
    cmd="$*"
    echo ${cmd} | tee ${result_dir}/cmdline.log
    eval ${cmd} 2>&1 | tee ${result_dir}/result.log

    [ -d logs ] && mv logs ${result_dir}
    [ -d .graph_dumps ] && mv .graph_dumps ${result_dir}/graph_dumps || mkdir -p ${result_dir}/graph_dumps
    echo $(find ${result_dir}/graph_dumps/ -maxdepth 1 -type f -name '*.pbtxt' | wc -l) graphs collected in ${result_dir}/graph_dumps
    mkdir -p ${result_dir}/graph_dumps/eager_graphs
    for x in $(seq 0 9); do mv ${x}*.pbtxt ${result_dir}/graph_dumps/eager_graphs 2>/dev/null; done
    echo $(find ${result_dir}/graph_dumps/eager_graphs/ -maxdepth 1 -type f -name '*.pbtxt' | wc -l) eager graphs collected in ${result_dir}/graph_dumps/eager_graphs
    cp *.py  ${result_dir}
    cp *.log  ${result_dir} 2>/dev/null
    cp $0 ${result_dir} 2>/dev/null
    chmod -R 777 ${result_dir}
    echo Results collected in ${result_dir}. Size $(du -sh ${result_dir})
}

export HABANA_LOGS=$(pwd)/logs

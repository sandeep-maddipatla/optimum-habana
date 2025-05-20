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
    mv *.pbtxt  ${result_dir}/graph_dumps 2>/dev/null
    cp *.py  ${result_dir}
    cp *.log  ${result_dir} 2>/dev/null
    echo $0 ${result_dir} 2>/dev/null
    cp $0 ${result_dir} 2>/dev/null
    chmod -R 777 ${result_dir}
}

export HABANA_LOGS=$(pwd)/logs

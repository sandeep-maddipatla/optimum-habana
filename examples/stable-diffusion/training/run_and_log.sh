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

prep_cpu()
{
    echo enable cpu performance governor
    #expects linux-tools-common, linux-tools-generic packages installed for cpupower tool
    CPU_POWER_TOOL_PATH=`find /usr/lib -executable -name cpupower | head -n 1`
    export cpu_count=0
    [ ! -x ${CPU_POWER_TOOL_PATH} ] && echo "cpupower tool not found" && exit
    echo Found tool at $CPU_POWER_TOOL_PATH
    cpu_count=`${CPU_POWER_TOOL_PATH} frequency-set --governor performance | wc -l`
    echo Set Frequency for ${cpu_count} CPUs
}

run() {
    tag=result
    result_dir=$(make_unique_dir ${tag})
    echo Using ${result_dir}

    pkill hl-smi
    hl-smi -l 1 -Q "timestamp,name,bus_id,driver_version,temperature.aip,utilization.aip,memory.total,memory.free,memory.used,pcie.link.gen.max,pcie.link.gen.current,pcie.link.width.max" -f csv > ${result_dir}/hlsmi.csv 2>&1 &
    hlsmi_pid=$(echo $!)
    echo Running hl-smi as background process $hlsmi_pid

    if [ "${HABANA_PROFILE}" == "1" ]; then
      echo HABANA_PROFILE=${HABANA_PROFILE}
      hl-prof-config -e off --hw-trace off
      hl-prof-config -e off --phase=device-acq -b 256 --skipParse On
    fi

    refresh_interval=1
    pkill mpstat
    mpstat -P ALL ${refresh_interval} > ${result_dir}/mpstat.log &
    mpstat_pid=$(echo $!)

    pkill free
    free --seconds ${refresh_interval} --mega --total --wide > ${result_dir}/free.log &
    free_pid=$(echo $!)

    pkill vmstat
    vmstat 1 > ${result_dir}/vmstat.log &
    vmstat_id=$(echo $!)

    # Actual run
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
    mv metricslog.json ${result_dir} 2>/dev/null
    mv *.hltv ${result_dir}
    chmod -R 777 ${result_dir}

    pkill hl-smi
    pkill mpstat
    pkill vmstat

    [ -d profile_logs ] && echo "Sleep 5s to ensure profiles are collected" && sleep 5
    [ -d profile_logs ] && mv profile_logs ${result_dir}
    echo Results collected in ${result_dir}. Size $(du -sh ${result_dir})
}

export HABANA_LOGS=$(pwd)/logs

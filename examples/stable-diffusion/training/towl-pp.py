import towl.user as tu
import matplotlib
import itables

scenario = tu.Scenario('/root/optimum-habana/examples/stable-diffusion/training/result_28_docker-304_logs_towl/towl_post_process/db/mydb')
print('Scenario global time range:', scenario.global_event_timerange)

global_view = scenario.make_view(scenario.global_event_timerange)
# shortcut:
global_view = scenario.make_global_view()

global_memory_usage_df = global_view.query_memory_usage()
print(type(global_memory_usage_df))
itables.show(global_memory_usage_df)

tu.plots.plot_memory_usage(global_view)
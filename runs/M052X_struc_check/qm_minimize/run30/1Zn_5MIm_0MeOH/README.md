old_initial_run is based on a old MOL.xyz, which has a non-converged gradient in ORCA's criteria. old_restart_run, old_another_restart_run are then based on the orc_job.xyz of old_initial_run

ORCA seems to have bug for optimizating this structure--it lead to SCF problems in later optimization iterations.

I have to use use a more strict threshold, and rerun the dlfind optimization in /home/zbai29/data/qmmm_test/react_benchmark/M052X_struc/qm_minimize/run30/1Zn_5MIm_0MeOH and copy its min.xyz as the updated MOL.xyz here. The result showed that MOL.xyz just meet the ORCA's gradient criteria for convergence. So I used it for min.xyz
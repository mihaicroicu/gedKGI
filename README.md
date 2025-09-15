# gedKGI

This is the KGI Gaussian Point imputation package used for creating a MSI multiple imputted dataset of UCDP GED based on Croicu (2025). There are two versions of the codebase, one for local use on a standard machine, and one for use on the Uppmax (rackham) cluster. 

The regular version of gedKGI will **not** work on Uppmax: the regular version depends on real-time API calls to UCDP, on the client-server nature of ingester3 and on a local cache buffer to speed client-server interactions and preempt some repeated compute operations. These **do not** function in a highly parallelized, high concurrency, batched environment like Uppmax, and will result in the script crashing in the best of cases, and ingester3 / UCDP API exposed to what is effectively a DDOS attack in the worst of cases (there are mechanisms in the cache cascade crashing the scripts before a DDOS will happen, but don't rely on this). In short **Don't do it**.

The Uppmax version is adapted for batch processing, separating out the parallel high concurrency work in separate scripts. Also, reliance on the ingester3 mechanisms, which are not built to scale to 5000+ concurrent calls, was eliminated completely, with a stand-alone minimal local reimplementation of the ingester3 API called "mingester". Mingester's API subset is compatible with the 1.9 branch of ingester3, but it is not under parallel development. 

Thusly, the two branches should *never* be merged, both are their own main.

# When to use Uppmax

To prepare a full GED version it is not practical to use a regular machine, a cluster is a good idea, as it takes ~450-500 core-hours on CPU and about 50-100 hours on a GPU. A monthly update is feasable on a regular machine, taking about 2 hours on an M1 TPU-equipped machine running Tensorflow on the Metal GPU/TPU.

## How to use it locally

The whole system is exposed through two command line applications, a command line application called `run2.py` which requries you to supply your own data and `run_ged.py` which fetches GED from the database. 
Running `python run_ged.py --help` will give you some help on how the process works. In short, the following flags exist:

`--month_id xxx` - Run for month_id xxx. If not specified, run for the latest month_id that has data in the UCDP Candidate API.
`--dyad_id xxx` - Run for only this dyad_id. If not specified, run for all dyads in given month_id
`--rebuild` - Rebuilds the whole system from month 1 to current month. Usefull for reloading a whole new GED. `month_id` is ignored in this case, but `dyad_id` is not - you can run this dyad by dyad.
`--seed xxx` - Set the seed.

The script is smart enough to fetch the correct datasets from UCDP and collate them correctly.


## How to use it on Uppmax

The procedure is slightly involved due to the batched nature of Uppmax, but the steps are simple enough.

1. Download the `uppmax` branch of the repo. I will assume you have it in  `~/gedKGI/`
2. Set up a symlink to the project directory containing the data storage required for KGI (i.e. the GADM geopackage(s)). These currently live at `/proj/uppstore2019113/mihai`. This means you should do `ln -s /proj/uppstore2019113/mihai ~/gedKGI/storage`
3. Install the `conda` module and the `gpp` environment.
4. Ask for a four hour single core interactive session. This is not techically needed, but it is good form, since you will run the data fetcher, which is a resource intensive session: `interactive -A snic2022-5-587 -n 1 -t 4:00:00`
5. After the interactive session started, activate the gpp environment with `conda activate gpp`.
6. Run `batch_maker.py`. This will fetch the current data from UCDP, collate the full and candidates data correctly, and create the required batch jobs (`runners`) that you will then submit to the batch system. Once the data is fetched once, it is stored - if you need to rerun `batch_maker.py` in the same month, you can do it from the login node, it rarely takes more than 1--2 minutes.
7. When 6. is done go into the `batcher` directory and run `bash register.bash`. Wait until all the jobs are registered, and come back in about 12 hours to check that all are done. Exit and log out.
8. Check you have no more `kgi` jobs running using `jobinfo -u USER`. If it's still running wait some more. If no more jobs are running, continue below.
9. Run `check_done.py` (you can do it in either the login or interactive node). If it prints `[OK]` and exits, you are ready for the next step. Otherwise, you should check the logs for the failed dyads. 
10. If `done` execute `python make_datasests.py`.

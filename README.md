# viewsKGI

This is the KGI Gaussian Point imputation package used for creating a MSI multiple imputted dataset of UCDP GED based on Croicu (2023) and ingesting the resulting dataset in VIEWS3. There are two versions of the codebase, one for local use on a standard machine, and one for use on the Uppmax (rackham) cluster. 

The regular version of viewsKGI will **not** work on Uppmax: the regular version depends on real-time API calls to UCDP, on the client-server nature of ingester3 and on a local cache buffer to speed client-server interactions and preempt some repeated compute operations. These **do not** function in a highly parallelized, high concurrency, batched environment like Uppmax, and will result in the script crashing in the best of cases, and ingester3 / UCDP API exposed to what is effectively a DDOS attack in the worst of cases (there are mechanisms in the cache cascade crashing the scripts before a DDOS will happen, but don't rely on this). In short **Don't do it**.

The Uppmax version is adapted for batch processing, separating out the parallel high concurrency work in separate scripts. Also, reliance on the ingester3 mechanisms, which are not built to scale to 5000+ concurrent calls, was eliminated completely, with a stand-alone minimal local reimplementation of the ingester3 API called "mingester". Mingester's API subset is compatible with the 1.9 branch of ingester3, but it is not under parallel development. 

Thusly, the two branches should *never* be merged, both are their own main.

# When to use Uppmax

To prepare and ingest a full GED version it is not practical to use a regular machine, a cluster is a good idea, as it takes ~450-500 core-hours on CPU and about 50-100 hours on a GPU. A monthly update is feasable on a regular machine, taking about 2 hours on an M1 TPU-equipped machine running Tensorflow on the Metal GPU/TPU.

## How to use it on Uppmax

The procedure is slightly involved due to the batched nature of Uppmax, but the steps are simple enough.

1. Download the `uppmax` branch of the repo.

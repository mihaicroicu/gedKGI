import click


@click.command()
@click.option('--iter', default=1, help='MC iteration to run')
@click.option('--kgi', default='0.2', help='KGI percentage')
@click.option('--bias', default='0.1', help='Spatial bias for KGI production')
@click.option('--seed', default=42, help='NP and TF Seed')
def hello(iter, kgi, bias, seed):
    return iter, kgi, bias, seed

if __name__ == '__main__':
    print("MUIE PULA")
    iter,kgi,bias,seed = hello()
    print (f"{iter=}, {kgi=}, {bias=}, {seed=}") 
#print(f"Running with {iter=} for level of {kgi=} and {bias=}")

import pandas as pd
import lithops
from phe import paillier
import pathlib
import matplotlib.pyplot as plt
import itertools
import time

bucket_name = lithops.Storage().bucket
print(f"> Bucket name: {bucket_name}")
key = 'dataframe'

def map(indexes, storage):
    subkeys = []
    for id in indexes:
        # Get the dataset from the storage backend (less than 4MiB to avoid MemoryError (IBM Cloud))
        df = pd.read_json(storage.get_object(bucket_name, f"{key}_{id}", stream=True), orient='split')
        # Compute the sum of the cost column for each customer for the current chunk
        df = df.groupby("customer_id", as_index=False).sum("cost")
        # Save the result to the storage backend
        subkey = f"map_{key}_{id}"
        storage.put_object(bucket_name, subkey, df.to_json(orient='split'))
        subkeys.append(subkey)
    return subkeys

def reduce(results, storage):
    files = list(itertools.chain(*results))
    # Read the files and concatenate them into a single dataframe (should be less than the 4MiB limit after the map phase)
    df = pd.concat([pd.read_json(storage.get_object(bucket_name, f, stream=True), orient='split') for f in files])
    # Compute the sum of the cost column for each customer for the entire dataset
    df = df.groupby("customer_id", as_index=False).sum("cost")
    return df

def get_mib_size(df):
    return df.memory_usage(index=True, deep=True).sum() / 1024 / 1024

def execute(dataset, number_workers, max_mib_chunk_size=4):
    df = pd.read_csv(dataset)

    storage = lithops.Storage()

    # Keep only the customer_id and cost columns
    df = df[['customer_id', 'cost']]

    row_size = df.shape[0]
    chunk_size = row_size // number_workers
    workload_for_each_worker = 1
    add_last_chunk = False

    # If df size of one chunk in MiB is > 4MiB,
    # we need to split the dataset into more chunks that will be processed by the same number of workers previously defined
    recompute_chunk_size = False
    for i in range(0, chunk_size-1, chunk_size):
        if get_mib_size(df.iloc[i:i+chunk_size]) > max_mib_chunk_size:
            recompute_chunk_size = True
            break
    if recompute_chunk_size:
        print(f"> /!\\ Chunk size is too big for the number of workers we have (>{max_mib_chunk_size}MiB), splitting the dataset into more chunks...")
        new_chunk_size = chunk_size
        done = False
        while not done:
            new_chunk_size = new_chunk_size // 2
            # Check if each new df chunk size is under 4MiB
            done = True
            for i in range(0, new_chunk_size-1, new_chunk_size):
                if get_mib_size(df.iloc[i:i+new_chunk_size]) > max_mib_chunk_size:
                    done = False
                    break
                
        workload_for_each_worker = chunk_size // new_chunk_size
        print(f"> New chunk row size: \t\t\t\t{new_chunk_size} rows/chunk")

        # If the number of workers is not a multiple of the new chunk size, we need to add a last chunk to the iterdata
        if chunk_size % new_chunk_size != 0:
            add_last_chunk = True
        chunk_size = new_chunk_size

    print(f"> Workload for each worker: \t\t\t{workload_for_each_worker} chunks/worker")

    # Encrypt the cost column using the Paillier cryptosystem
    #public_key, private_key = paillier.generate_paillier_keypair()
    #df['cost'] = df['cost'].apply(lambda x: public_key.encrypt(x))

    # Get array of array of indexes for each worker and send the corresponding chunks to the storage backend
    indexes_iterdata = []
    for n in range(0, number_workers):
        chunk_ids = []
        for w in range(0, workload_for_each_worker):
            chunk_id = w*number_workers + n
            chunk_ids.append(chunk_id)
            print(f"> Uploading Chunk {chunk_id} to the storage backend...")
            boundary = chunk_id*chunk_size
            storage.put_object(bucket_name, f"{key}_{chunk_id}", df.iloc[boundary:boundary+chunk_size].to_json(orient='split'))
            # Add the last chunk if needed
            if n == number_workers and w == workload_for_each_worker and add_last_chunk:
                print(f"> Uploading Chunk {chunk_id+1} to the storage backend...")
                storage.put_object(bucket_name, f"{key}_{chunk_id+1}", df.iloc[(chunk_id+1)*chunk_size:].to_json(orient='split'))
                chunk_ids.append(chunk_id+1)
        indexes_iterdata.append(chunk_ids)

    fexec = lithops.FunctionExecutor()

    print(f"> Repartition of the chunks for each worker (map_iterdata): \t\t\t{indexes_iterdata}")

    # Start the timer
    start_time = time.time()

    # Perform the parallel reduction
    futures = fexec.map_reduce(map, indexes_iterdata, reduce)
    result_df = fexec.get_result(futures)

    # Get the workers' execution time
    execution_time = time.time() - start_time
    
    # Decrypt the cost
    #result_df['cost'] = result_df['cost'].apply(lambda x: private_key.decrypt(x))

    # Filter the entries to find the customers who have spent more than $5k
    result_df = result_df[result_df['cost'] > 5000].reset_index(drop=True)

    # Print the result
    print("> RESULTS: customers who have spent more than $5k")
    print(result_df)

    # Return the customer_id of the customers who have spent more than $5k and the time it took to compute the result
    return result_df['customer_id'], execution_time

def main():
    df_metrics = pd.DataFrame(columns=['dataset', 'size', 'number_workers', 'execution_time', 'speedup'])
    # For each dataset, we process it and measure the execution time of map_reduce function between 1 worker and n workers
    for dataset in [pathlib.Path('datasets/store_sales_SF0_5.csv'), pathlib.Path('datasets/store_sales_SF0_1.csv'), pathlib.Path('datasets/store_sales_SF1.csv')]:
        size = dataset.stat().st_size / 1024 / 1024
        print('----------------------------------------')
        print(f'> Dataset: {dataset} | Size of the dataset: {size} MiB | Number of workers: 1')
        _, execution_time_one_worker = execute(dataset, 1)
        print(f'> Execution time: {execution_time_one_worker} seconds')
        print(f'> Speedup: 1')
        df_metrics = pd.concat([df_metrics, pd.DataFrame({'dataset': [dataset], 'size': [size], 'number_workers': [1], 'execution_time': [execution_time_one_worker], 'speedup': [1]})], ignore_index=True)
        print()
        for number_workers in [2, 4, 8, 16, 32]:
            print('----------------------------------------')
            print(f'> Dataset: {dataset} | Size of the dataset: {size} MiB | Number of workers: {number_workers}')
            _, execution_time = execute(dataset, number_workers)
            print(f'> Execution time: {execution_time} seconds')
            speedup = execution_time_one_worker / execution_time
            print(f'> Speedup: {speedup}')
            df_metrics = pd.concat([df_metrics, pd.DataFrame({'dataset': [dataset], 'size': [size], 'number_workers': [number_workers], 'execution_time': [execution_time], 'speedup': [speedup]})], ignore_index=True)
            print()

    # Save the metrics to a csv file
    df_metrics.to_csv('metrics/metrics.csv', index=False)

    # Plot the metrics
    for dataset in df_metrics['dataset'].unique():
        df = df_metrics[df_metrics['dataset'] == dataset]
        plt.plot(df['number_workers'], df['speedup'], label=dataset)
        plt.xlabel('Number of workers')
        plt.ylabel('Speedup')
        plt.title(dataset)
        plt.legend()
        plt.show()

if __name__ == '__main__':
    main()
import pandas as pd
import lithops
from phe import paillier
import pathlib
import matplotlib.pyplot as plt

bucket_name = 'bucket-rm'
key = 'dataframe'

def map(id, index, chunk_size, key, storage):
    # Get the dataset from the storage backend
    df = pd.read_json(storage.get_object(bucket_name, key, stream=True), orient='split')
    # Filter the dataset to keep only the rows that are assigned to the current worker
    df = df.iloc[index:index+chunk_size]
    # Compute the sum of the cost column for each customer for the current chunk
    df = df.groupby("customer_id", as_index=False).sum("cost")
    # Save the result to the storage backend
    storage.put_object(bucket_name, f"{key}_{id}", df.to_json(orient='split'))
    return f"{key}_{id}"

def reduce(results, storage):
    files = results
    # Read the files and concatenate them into a single dataframe
    df = pd.concat([pd.read_json(storage.get_object(bucket_name, f, stream=True), orient='split') for f in files])
    # Compute the sum of the cost column for each customer for the entire dataset
    df = df.groupby("customer_id", as_index=False).sum("cost")
    return df

def execute(dataset, number_workers):
    df = pd.read_csv(dataset)

    storage = lithops.Storage()

    # Keep only the customer_id and cost columns
    df = df[['customer_id', 'cost']]

    row_size = df.shape[0]
    chunk_size = row_size // number_workers

    # Get array of indexes for each worker
    indexes_iterdata = [i * chunk_size for i in range(number_workers + 1)]

    # Encrypt the cost column using the Paillier cryptosystem
    #public_key, private_key = paillier.generate_paillier_keypair()
    #df['cost'] = df['cost'].apply(lambda x: public_key.encrypt(x))

    # Send the dataset to the storage backend
    storage.put_object(bucket_name, key, df.to_json(orient='split'))

    fexec = lithops.FunctionExecutor()

    # Perform the parallel reduction
    futures = fexec.map_reduce(map, indexes_iterdata, reduce, extra_args=(chunk_size, key))
    result_df = fexec.get_result(futures)

    # Add each worker's execution time to the total worker execution time
    execution_time = sum([f.stats['worker_exec_time'] for f in futures])
    
    # Decrypt the cost
    #result_df['cost'] = result_df['cost'].apply(lambda x: private_key.decrypt(x))

    # Filter the entries to find the customers who have spent more than $5k
    result_df = result_df[result_df['cost'] > 5000].reset_index(drop=True)

    # Print the result
    print(result_df)

    # Return the customer_id of the customers who have spent more than $5k and the time it took to compute the result
    return result_df['customer_id'], execution_time

def main():
    df_metrics = pd.DataFrame(columns=['dataset', 'size', 'number_workers', 'execution_time', 'speedup'])
    # For each dataset, we process it and measure the execution time of map_reduce function between 1 worker and n workers
    for dataset in [pathlib.Path('datasets/store_sales_SF0_1.csv'), pathlib.Path('datasets/store_sales_SF0_5.csv'), pathlib.Path('datasets/store_sales_SF1.csv')]:
        size = dataset.stat().st_size / 1024 / 1024
        print('----------------------------------------')
        print(f'Dataset: {dataset} | Size of the dataset: {size} MiB | Number of workers: 1')
        _, execution_time_one_worker = execute(dataset, 1)
        print(f'Execution time: {execution_time_one_worker} seconds')
        print(f'Speedup: 1')
        df_metrics = pd.concat([df_metrics, pd.DataFrame({'dataset': [dataset], 'size': [size], 'number_workers': [1], 'execution_time': [execution_time_one_worker], 'speedup': [1]})], ignore_index=True)
        print()
        for number_workers in [2, 4, 8, 16, 32]:
            print('----------------------------------------')
            print(f'Dataset: {dataset} | Size of the dataset: {size} MiB | Number of workers: {number_workers}')
            _, execution_time = execute(dataset, number_workers)
            print(f'Execution time: {execution_time} seconds')
            speedup = execution_time_one_worker / execution_time
            print(f'Speedup: {speedup}')
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
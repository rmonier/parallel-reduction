import pandas as pd
import lithops
from phe import paillier
from phe.util import int_to_base64, base64_to_int
import pathlib
import matplotlib.pyplot as plt
import itertools
import time
import json
import argparse

bucket_name = lithops.Storage().bucket
print(f"> Bucket name: {bucket_name}")
key = 'dataframe'

def groupby_sum_for_cipher(df, group_col, sum_col):
    # We are using a custom groupby sum function because the default one is not working with Paillier cipher
    # We need to add the encrypted costs with the + operator directly
    result = {}
    for _, row in df.iterrows():
        group = row[group_col]
        cost = row[sum_col]
        if group in result:
            result[group] += cost
        else:
            result[group] = cost
    return pd.DataFrame({group_col: list(result.keys()), sum_col: list(result.values())})

def map(indexes, encryption, n_public_key, storage):
    encrypt_data = {"encryption": encryption, "n_public_key": n_public_key}
    if encrypt_data["encryption"]:
        import phe
        public_key = paillier.PaillierPublicKey(n=base64_to_int(encrypt_data['n_public_key']))
    subkeys = []
    for id in indexes:
        # Get the dataset from the storage backend (less than 4MiB to avoid MemoryError (IBM Cloud))
        df = pd.read_json(storage.get_object(bucket_name, f"{key}_{id}", stream=True), orient='split', dtype={'customer_id': int, 'cost': str} if encrypt_data["encryption"] else None)
        if encrypt_data["encryption"]:
            # Get the Paillier cipher for the cost column
            df['cost'] = df['cost'].apply(lambda x: decode_cipher_json(x, public_key))
        # Compute the sum of the cost column for each customer for the current chunk
        if not encrypt_data["encryption"]:
            df = df.groupby("customer_id", as_index=False).sum("cost")
        else:
            df = groupby_sum_for_cipher(df, "customer_id", "cost")
            df['cost'] = df['cost'].apply(lambda x: get_json_for_cipher(x))
        # Save the result to the storage backend
        subkey = f"map_{key}_{id}"
        storage.put_object(bucket_name, subkey, df.to_json(orient='split'))
        subkeys.append(subkey)
    return {"encrypt_data": encrypt_data, "subkeys": subkeys}

def reduce(results, storage):
    encrypt_data = results[0]["encrypt_data"]  # We assume that all the workers will have the same encryption data
    subkeys = [item['subkeys'] for item in results]  # We extract the subkeys from the results
    if encrypt_data["encryption"]:
        import phe
        public_key = paillier.PaillierPublicKey(n=base64_to_int(encrypt_data["n_public_key"]))
    files = list(itertools.chain(*subkeys))
    # Read the files and concatenate them into a single dataframe (should be less than the 4MiB limit after the map phase)
    df = pd.concat([pd.read_json(storage.get_object(bucket_name, f, stream=True), orient='split') for f in files])
    if encrypt_data["encryption"]:
        # Get the Paillier cipher for the cost column
        df['cost'] = df['cost'].apply(lambda x: decode_cipher_json(x, public_key))
    # Compute the sum of the cost column for each customer for the entire dataset
    if not encrypt_data["encryption"]:
        df = df.groupby("customer_id", as_index=False).sum("cost")
    else:
        df = groupby_sum_for_cipher(df, "customer_id", "cost")
        df['cost'] = df['cost'].apply(lambda x: get_json_for_cipher(x))
    return df

def get_mib_size(df):
    return df.memory_usage(index=True, deep=True).sum() / 1024 / 1024

def get_json_for_cipher(cipher):
    return json.dumps((int_to_base64(cipher.ciphertext()), cipher.exponent))

def encode_cipher_json(number, public_key):
    enc = public_key.encrypt(number, precision=1e-2)
    return get_json_for_cipher(enc)

def decode_cipher_json(value, public_key):
    value = json.loads(value)
    return paillier.EncryptedNumber(public_key, ciphertext=base64_to_int(value[0]), exponent=int(value[1]))

def execute(dataset, number_workers, max_mib_chunk_size=4, encryption=False):
    df = pd.read_csv(dataset)

    storage = lithops.Storage()

    # Keep only the customer_id and cost columns
    df = df[['customer_id', 'cost']]

    row_size = df.shape[0]
    chunk_size = row_size // number_workers
    if chunk_size <= 0:
        chunk_size = 1
    workload_for_each_worker = 1
    add_last_chunk = False

    print("> Paillier Encryption: " + ("ON" if encryption else "OFF"))

    if encryption:
        # Generate the public and private keys for the Paillier cryptosystem
        print("> Generating public and private keys for the Paillier cryptosystem...")
        public_key, private_key = paillier.generate_paillier_keypair(n_length=128)  # We should use a 2048-bit key to be secure, but as it takes a long time to encrypt, we will use a 128-bit key for this project
        
        # Encrypt the cost column using the Paillier cryptosystem
        encrypt_str = "> Encrypting the cost column (can take a long time)..."
        print(encrypt_str, end="", flush=True)
        cost_size = df['cost'].size
        current_cost = 0
        print(f"\r{encrypt_str} {(current_cost/cost_size*100):.2f} %", end="", flush=True)
        def encrypt_cost(x, public_key):
            enc = encode_cipher_json(x, public_key)
            nonlocal current_cost
            current_cost += 1
            print(f"\r{encrypt_str} {(current_cost/cost_size*100):.2f} %", end="", flush=True)
            return enc

        df['cost'] = df['cost'].apply(lambda x: encrypt_cost(x, public_key))
        print("")

    # If df size of one chunk in MiB is > 4MiB,
    # we need to split the dataset into more chunks that will be processed by the same number of workers previously defined
    print("> Checking the chunk size...")
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

    base64_key = "" if not encryption else int_to_base64(public_key.n)

    # Start the timer
    start_time = time.time()

    # Perform the parallel reduction
    futures = fexec.map_reduce(map, indexes_iterdata, reduce, extra_args=(encryption, base64_key))
    result_df = fexec.get_result(futures)

    # Get the workers' execution time
    execution_time = time.time() - start_time
    
    if encryption:
        # Decrypt the cost
        decrypt_str = "> Decrypting the cost column..."
        print(decrypt_str, end="", flush=True)
        cost_size = result_df['cost'].size
        current_cost = 0
        print(f"\r{decrypt_str} {(current_cost/cost_size*100):.2f} %", end="", flush=True)
        def decrypt_cost(x, public_key, private_key):
            dec = decode_cipher_json(x, public_key)
            dec = round(private_key.decrypt(dec), 2)
            nonlocal current_cost
            current_cost += 1
            print(f"\r{decrypt_str} {(current_cost/cost_size*100):.2f} %", end="", flush=True)
            return dec

        result_df['cost'] = result_df['cost'].apply(lambda x: decrypt_cost(x, public_key, private_key))
        print("")

    # Filter the entries to find the customers who have spent more than $5k
    result_df = result_df[result_df['cost'] > 5000].reset_index(drop=True)

    # Print the result
    print("> RESULTS: customers who have spent more than $5k")
    print(result_df)

    # Return the customer_id of the customers who have spent more than $5k and the time it took to compute the result
    return result_df['customer_id'], execution_time

def main():
    # Get the arguments from the command line
    parser = argparse.ArgumentParser(
                    prog = 'Parallel Reduction',
                    description = 'Parallel Reduction using Lithops and MapReduce, with or without Paillier encryption')
    parser.add_argument('--encrypt', action='store_true', default=False,
                        help='If set, the cost column will be encrypted using the Paillier cryptosystem before performing the parallel reduction.')
    args = parser.parse_args()

    df_metrics = pd.DataFrame(columns=['dataset', 'size', 'number_workers', 'execution_time', 'speedup'])
    # For each dataset, we process it and measure the execution time of map_reduce function between 1 worker and n workers
    for dataset in [pathlib.Path('datasets/store_sales_SF0_1.csv'), pathlib.Path('datasets/store_sales_SF0_5.csv'), pathlib.Path('datasets/store_sales_SF1.csv')]:
        size = dataset.stat().st_size / 1024 / 1024
        print('----------------------------------------')
        print(f'> Dataset: {dataset} | Size of the dataset: {size} MiB | Number of workers: 1')
        _, execution_time_one_worker = execute(dataset, 1, encryption=args.encrypt)
        print(f'> Execution time: {execution_time_one_worker} seconds')
        print(f'> Speedup: 1')
        df_metrics = pd.concat([df_metrics, pd.DataFrame({'dataset': [dataset], 'size': [size], 'number_workers': [1], 'execution_time': [execution_time_one_worker], 'speedup': [1]})], ignore_index=True)
        print()
        for number_workers in [2, 4, 8, 16, 32]:
            print('----------------------------------------')
            print(f'> Dataset: {dataset} | Size of the dataset: {size} MiB | Number of workers: {number_workers}')
            _, execution_time = execute(dataset, number_workers, encryption=args.encrypt)
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
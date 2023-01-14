import pandas as pd
import lithops
from phe import paillier

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

def main():
    number_workers = 6

    df = pd.read_csv('datasets/store_sales_SF0_1.csv')
    df = pd.concat([df, pd.read_csv('datasets/store_sales_SF0_5.csv')])
    df = pd.concat([df, pd.read_csv('datasets/store_sales_SF1.csv')])

    storage = lithops.Storage()

    # Keep only the customer_id and cost columns
    df = df[['customer_id', 'cost']]
    print('Size of the dataset: {} MiB'.format(df.memory_usage(index=True).sum() / 1024**2))

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
    
    # Decrypt the cost
    #result_df['cost'] = result_df['cost'].apply(lambda x: private_key.decrypt(x))

    # Filter the entries to find the customers who have spent more than $5k
    result_df = result_df[result_df['cost'] > 5000].reset_index(drop=True)

    # Print the final results
    print(result_df)

if __name__ == '__main__':
    main()
import boto3

if __name__ == "__main__":
    s3 = boto3.client('s3')
    bucket_name = 'your-bucket-name'  # Replace with your bucket name
    prefixes = ['pagerank/', 'community/', 'dijkstra']

    for prefix in prefixes:
        paginator = s3.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
            if 'Contents' in page:
                objects = [{'Key': obj['Key']} for obj in page['Contents']]
                s3.delete_objects(Bucket=bucket_name, Delete={'Objects': objects})

    print("Clean up (graphs) completed.")
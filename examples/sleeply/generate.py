#generate sleepy

import pickle
from lithops import Storage

BUCKET = "your-bucket-name"     #
STAGE_1_DURATION = 10           # seconds
STAGE_2_DURATION = 15  
STAGE_3_DURATION = 20  
STAGE_1_SIZE = 128              # bytes  
STAGE_2_SIZE = 256
STAGE_3_SIZE = 512

def gen_sleep():
    storage = Storage()
    storage.create_bucket(BUCKET)
    
    stages = [
        {"duration": STAGE_1_DURATION, "size": STAGE_1_SIZE},
        {"duration": STAGE_2_DURATION, "size": STAGE_2_SIZE},
        {"duration": STAGE_3_DURATION, "size": STAGE_3_SIZE}
    ]
    
    for i, stage in enumerate(stages):
        data = {
            "stage": i + 1,
            "duration": stage["duration"],
            "size": stage["size"]
        }
        storage.put_object(BUCKET, f"sleeply/sleep_stage{i+1}", pickle.dumps(data))
        del data


if __name__ == "__main__":
    gen_sleep()
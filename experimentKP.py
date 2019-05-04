import pandas as pd

from knapsack import optimizeOnePlusOne
from glob import iglob
from model import TTSP

def run():
    #algorithms = [optimizeOnePlusOne]
    df = pd.DataFrame(columns=['filename', 'algorithm', 'iterations', 'solution', 'time',
                               'kp_capacity', 'item_number'])
    for file in iglob('data/**.ttp'):
        ttsp = TTSP(file)
        df = df.append({'filename': file, 'kp_capacity': ttsp.knapsack_capacity,
                   'item_number':ttsp.item_num}, ignore_index=True)




if __name__ == '__main__':
    run()
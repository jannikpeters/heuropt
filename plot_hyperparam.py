from glob import iglob
import pandas as pd
import matplotlib.pyplot as plt

def load_file_names():
    names = ['a280_n279', 'a280_n2790','a280_n1395',
      'fnl4461_n4460', 'fnl4461_n22300', 'fnl4461_n44600',
     'pla33810_n33809', 'pla33810_n169045', 'pla33810_n338090']

    rows = []
    for problem_name in names:
        for file_path in iglob('gecco_solutions/'+problem_name+'/*'):
            file = file_path.split('/')[-1]
            parts = file.split('_')
            algo = parts[2]
            profit = int(parts[3][1:])
            c = float(parts[4][1:])
            rows.append({'problem_name': problem_name,
                   'algo': algo,
                   'profit': profit,
                   'c': c})
    df = pd.DataFrame(rows)
    return df


if __name__ == '__main__':
    df = load_file_names()
    probs = df.problem_name.unique()
    for prob in probs:
        algos = ['greed', 'rev']
        for alg in algos:
            a_prob = df[(df.problem_name == prob) & (df.algo == alg)].groupby('c').max()
            a_prob.profit.plot(label=alg)
        plt.title(prob)
        plt.ylabel('profit')
        plt.legend()
        plt.show()

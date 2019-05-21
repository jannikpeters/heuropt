from glob import iglob


def load_file_names():
    names = ['a280_n279', 'a280_n2790','a280_n1395',
      'fnl4461_n4460', 'fnl4461_n22300', 'fnl4461_n44600',
     'pla33810_n33809', 'pla33810_n169045', 'pla33810_n338090']

    dict = {}
    for problem_name in names:
        dict[problem_name] = []
        for file_path in iglob('gecco_solutions/'+problem_name+'/*'):
            file = file_path.split('/')[-1]
            parts = file.split('_')
            type = parts[2]
            profit = int(parts[3][1:])
            c = float(parts[4][1:])
            print('hi')


if __name__ == '__main__':
    load_file_names()
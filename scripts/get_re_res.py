import sys

sys.path.append('/data1/deepN/highie')

from eval_relation import eval_relation

import os


# dirs = [ 'log/ace05-R/baseline-ident2_20220504_025934', 'log/ace05-R/baseline-ident2_20220504_025947','log/ace05-R/baseline-ident2_20220504_030000']
dirs = ['log/scierc/sib+cop-noshare_20220504_132618', 'log/scierc/sib+cop-noshare_20220504_132638', 'log/scierc/sib+cop-noshare_20220504_132652']

for dir_name in dirs:

  # test_scores = eval_relation('data/ace05-R/test.albert.json',os.path.join(dir_name,'result.test.json'))

  # dev_scores = eval_relation('data/ace05-R/dev.albert.json',os.path.join(dir_name,'result.dev.json'))

  test_scores = eval_relation('data/scierc/test.oneie.json',os.path.join(dir_name,'result.test.json'))

  dev_scores = eval_relation('data/scierc/dev.oneie.json',os.path.join(dir_name,'result.dev.json'))


  test_scores.extend(dev_scores)

  # print("-"*120)

  print(','.join(test_scores))

  # print("-"*120)


print('\n')
for dir_name in dirs:

  # test_scores = eval_relation('data/ace05-R/test.albert.json',os.path.join(dir_name,'final.result.test.json'))

  # dev_scores = eval_relation('data/ace05-R/dev.albert.json',os.path.join(dir_name,'final.result.dev.json'))

  test_scores = eval_relation('data/scierc/test.oneie.json',os.path.join(dir_name,'final.result.test.json'))

  dev_scores = eval_relation('data/scierc/dev.oneie.json',os.path.join(dir_name,'final.result.dev.json'))


  test_scores.extend(dev_scores)

  # print("-"*120)

  print(','.join(test_scores))

  # print("-"*120)
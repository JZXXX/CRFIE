import sys

sys.path.append('/data1/zhangsc/deepN/highIE-event')

from eval_ee import eval_ee

import os

dirs = ['log/ace05-EN/e+r-sib_20220624_134655', 'log/ace05-EN/e+r-sib_20220624_134707','log/ace05-EN/e+r-sib_20220624_134720']
# dirs = ['log/dygie/cop-noshare_20220621_123815','log/dygie/cop-noshare_20220621_123830']
res = {}
i = 0
for dir_name in dirs:

  # test_scores = eval_ee('data/dygie/test.oneie.json',os.path.join(dir_name,'result.test.json'))

  # dev_scores = eval_ee('data/dygie/dev.oneie.json',os.path.join(dir_name,'result.dev.json'))

  test_scores = eval_ee('data/ace05-EN/test.oneie.json',os.path.join(dir_name,'result.test.json'))

  dev_scores = eval_ee('data/ace05-EN/dev.oneie.json',os.path.join(dir_name,'result.dev.json'))

  print("test scores........................")

  print(str('%.2f'%(test_scores['entity']['f']*100))+','+str('%.2f'%(test_scores['trigger_id']['f']*100))+','+str('%.2f'%(test_scores['trigger']['f']*100))+','+str('%.2f'%(test_scores['role_id']['f']*100))

    +','+str('%.2f'%(test_scores['role']['f']*100))+','+str('%.2f'%(test_scores['relation']['f']*100))+',,'+
    str('%.2f'%(dev_scores['entity']['f']*100))+','+str('%.2f'%(dev_scores['trigger_id']['f']*100))+','+str('%.2f'%(dev_scores['trigger']['f']*100))+','+str('%.2f'%(dev_scores['role_id']['f']*100))

    +','+str('%.2f'%(dev_scores['role']['f']*100))+','+str('%.2f'%(dev_scores['relation']['f']*100)))

  res[i] = (str('%.2f'%(test_scores['entity']['f']*100))+','+str('%.2f'%(test_scores['trigger_id']['f']*100))+','+str('%.2f'%(test_scores['trigger']['f']*100))+','+str('%.2f'%(test_scores['role_id']['f']*100))

    +','+str('%.2f'%(test_scores['role']['f']*100))+','+str('%.2f'%(test_scores['relation']['f']*100))+',,'+
    str('%.2f'%(dev_scores['entity']['f']*100))+','+str('%.2f'%(dev_scores['trigger_id']['f']*100))+','+str('%.2f'%(dev_scores['trigger']['f']*100))+','+str('%.2f'%(dev_scores['role_id']['f']*100))

    +','+str('%.2f'%(dev_scores['role']['f']*100))+','+str('%.2f'%(dev_scores['relation']['f']*100)))
  # print("dev scores........................")

  # print(str('%.2f'%(dev_scores['entity']['f']*100))+','+str('%.2f'%(dev_scores['trigger_id']['f']*100))+','+str('%.2f'%(dev_scores['trigger']['f']*100))+','+str('%.2f'%(dev_scores['role_id']['f']*100))

  #   +','+str('%.2f'%(dev_scores['role']['f']*100))+','+str('%.2f'%(dev_scores['relation']['f']*100)))
  i+=1
print('\n')
res[i] = ''
i += 1
for dir_name in dirs:

  # test_scores = eval_ee('data/dygie/test.oneie.json',os.path.join(dir_name,'final.result.test.json'))

  # dev_scores = eval_ee('data/dygie/dev.oneie.json',os.path.join(dir_name,'final.result.dev.json'))
  test_scores = eval_ee('data/ace05-EN/test.oneie.json',os.path.join(dir_name,'final.result.test.json'))

  dev_scores = eval_ee('data/ace05-EN/dev.oneie.json',os.path.join(dir_name,'final.result.dev.json'))

  print("test scores........................")

  print(str('%.2f'%(test_scores['entity']['f']*100))+','+str('%.2f'%(test_scores['trigger_id']['f']*100))+','+str('%.2f'%(test_scores['trigger']['f']*100))+','+str('%.2f'%(test_scores['role_id']['f']*100))

    +','+str('%.2f'%(test_scores['role']['f']*100))+','+str('%.2f'%(test_scores['relation']['f']*100))+',,'
    +str('%.2f'%(dev_scores['entity']['f']*100))+','+str('%.2f'%(dev_scores['trigger_id']['f']*100))+','+str('%.2f'%(dev_scores['trigger']['f']*100))+','+str('%.2f'%(dev_scores['role_id']['f']*100))

    +','+str('%.2f'%(dev_scores['role']['f']*100))+','+str('%.2f'%(dev_scores['relation']['f']*100)))

  res[i] = (str('%.2f'%(test_scores['entity']['f']*100))+','+str('%.2f'%(test_scores['trigger_id']['f']*100))+','+str('%.2f'%(test_scores['trigger']['f']*100))+','+str('%.2f'%(test_scores['role_id']['f']*100))

    +','+str('%.2f'%(test_scores['role']['f']*100))+','+str('%.2f'%(test_scores['relation']['f']*100))+',,'
    +str('%.2f'%(dev_scores['entity']['f']*100))+','+str('%.2f'%(dev_scores['trigger_id']['f']*100))+','+str('%.2f'%(dev_scores['trigger']['f']*100))+','+str('%.2f'%(dev_scores['role_id']['f']*100))

    +','+str('%.2f'%(dev_scores['role']['f']*100))+','+str('%.2f'%(dev_scores['relation']['f']*100)))
  i += 1

  # print("dev scores........................")

  # print(str('%.2f'%(dev_scores['entity']['f']*100))+','+str('%.2f'%(dev_scores['trigger_id']['f']*100))+','+str('%.2f'%(dev_scores['trigger']['f']*100))+','+str('%.2f'%(dev_scores['role_id']['f']*100))

  #   +','+str('%.2f'%(dev_scores['role']['f']*100))+','+str('%.2f'%(dev_scores['relation']['f']*100)))
print('\n')
for r in res:
  print(res[r])
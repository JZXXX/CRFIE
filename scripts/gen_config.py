import json
from functools import reduce
import os
import time

def gen_config(template_path, *parameter_str):
    # parameter_str: 'tre_true', 'decop_300,100', 'iter_1,2,3',....
    with open(template_path, 'r', encoding='utf-8') as r:
        template_dict = json.load(r)
    short2key = {'tre': 'use_high_order_tre',
                 'sib': 'use_high_order_sibling',
                 'cop': 'use_high_order_coparent',
                 'ere': 'use_high_order_ere',
                 'er': 'use_high_order_er',
                 'decomp': 'decomp_size',
                 'iter': 'mfvi_iter',
                 'bs': 'batch_size',
                 'eve': 'event_classification',
                 'rel': 'relation_classification',
                 'ss': 'span_encoder_share',
                 'ts': 'token_encoder_share',
                 'reb':'rebatch'}

    write_dir = os.path.dirname(template_path)


    group = []
    paras = list(parameter_str)
    para_name = []
    for i in paras:
        pa, va = i.split('_')
        para_name.append(pa)
        va = list(va.split(','))
        para = []
        for n in va:
            para.append(pa+n)
        group.append(para)
    comb = lambda x : reduce(lambda x,y : [i+'_'+j for i in x for j in y], x)
    filename = comb(group)

    command_file = os.path.join(write_dir, 'command.txt')

    with open(command_file, 'a') as fw:
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        fw.write('Generate config in {}'.format(timestamp)+'\n')
        fw.write('\n')
        for fn in filename:
            fns = fn.split('_')
            pa_id = 0
            for i in fns:
                key = para_name[pa_id]
                value = i.replace(key, '')
                pa_id += 1
                if value == 'true':
                    template_dict[short2key[key]] = True
                elif value == 'false':
                    template_dict[short2key[key]] = False
                else:
                    template_dict[short2key[key]] = int(value)
                # template_dict[short2key[key]] = bool(value) if value in ['true','false'] else int(value)
            config_file = os.path.join(write_dir, fn+'.json')
            with open(config_file, 'w') as f:
                json.dump(template_dict, f, ensure_ascii=False, indent=4)
            sh_file = os.path.join(write_dir, fn+'.sh')

            nohup_out_file = 'nohup_out/{}'.format(os.path.join(write_dir.split('/')[-1], fn+'-{}.log'.format(i)))
            if not os.path.exists(os.path.dirname(nohup_out_file)):
                # breakpoint()
                os.makedirs(os.path.dirname(nohup_out_file))
            with open(sh_file, 'w') as fw1:
                for i in range(1,4):
                    fw1.write('CUDA_VISIBLE_DEVICES={} python train.py --config {}/{} >nohup_out/{}'.format(
                        filename.index(fn)%8, write_dir,fn+'.json', os.path.join(write_dir.split('/')[-1], fn+'-{}.log'.format(i))) + '\n')
            fw.write('nohup sh {} >1&'.format(sh_file) + '\n')
        fw.write('\n') 
# gen_config('config/dygie/baseline.json', 'tre_True', 'bs_10,32', 'iter_1,2,3')

if __name__ == '__main__':
    """"""
    import sys
    gen_config(sys.argv[1], *sys.argv[2:])




import os

def statistic_res(dir_name, files):

    with open(os.path.join(dir_name,'res_best.txt'), 'a') as fw:
        for file in files:
            with open(os.path.join(dir_name,file), 'r') as fr:
                lines = fr.readlines()
                try:
                    if len(lines[-2].strip().split(' ')) == 14:
                        fw.write(file+'\t'+'\t'.join(lines[-2].strip().split(' '))+'\n')
                    elif len(lines[-2].strip().split('\t')) == 7:
                        line = lines[-2].strip().split('\t')
                        line_ = ''
                        for i in line:
                            line_ += i.split(' ')[0]+'\t'+i.split(' ')[1]+'\t'
                        # breakpoint()
                        fw.write(file+'\t'+line_.strip()+'\n')
                    else:
                        print(file)
                        continue
                except:
                    print(file)
                    continue
    with open(os.path.join(dir_name,'res_final.txt'), 'a') as fw:
        for file in files:
            with open(os.path.join(dir_name,file), 'r') as fr:
                lines = fr.readlines()
                try:
                    if len(lines[-1].strip().split(' ')) == 14:
                        fw.write(file+'\t'+'\t'.join(lines[-1].strip().split(' '))+'\n')
                    elif len(lines[-1].strip().split('\t')) == 7:
                        line = lines[-1].strip().split('\t')
                        line_ = ''
                        for i in line:
                            line_ += i.split(' ')[0]+'\t'+i.split(' ')[1]+'\t'
                        # breakpoint()
                        fw.write(file+'\t'+line_.strip()+'\n')
                    else:
                        print(file)
                        continue
                except:
                    print(file)
                    continue





if __name__ == '__main__':
    """"""
    import sys
    dir_name = sys.argv[1]
    files = os.listdir(dir_name)
    files.sort()
    statistic_res(dir_name, files)
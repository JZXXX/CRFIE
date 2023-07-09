import matplotlib.pyplot as plt
n = 0
dev_f1 = []
test_f1 = []
total_loss = []
with open('better2.log', 'r') as fr:
    for line in fr:
        if line.startswith('total_loss:'):
            total_loss.append(float(line.strip().split(':')[-1]))
        if line.startswith('Role:'):
            n += 1
            if n%2 == 1:
                dev_f1.append(float(line.strip().split(':')[-1]))
            else:
                test_f1.append(float(line.strip().split(':')[-1]))

x = range(len(total_loss))
plt.plot(x,dev_f1)
plt.plot(x,test_f1)
# plt.plot(x,total_loss)
plt.show()


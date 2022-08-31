file_path = './PKLot/'
file_list = ['all', 'PUC_test', 'nottwodays', 'PUC_train', 'PUC', 'twodays', 'train']

for file in file_list:
    file_name = file_path + file + '.txt'
    print(file_name)
    
    pre = open(file_name, 'r')
    new = open('./'+file+'.txt', 'w')
    
    lines = pre.readlines()
    
    for line in lines:
        if "(2)" in line:
            print(line)
        else:
            new.write(line)
    pre.close()
    new.close()
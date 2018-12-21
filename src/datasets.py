#!/usr/bin/python
#coding:utf-8

import json
import pickle

class Datesets(object):

    def __init__(self):
        pass

    def get_sample_datesets(self, data_path, task, sample_nums=1000, is_sampls=False):
        """
        以12小时为单位存储
        :param train_path:训练数据的路径
        :param task:任务：'mortality/readmission/diagnosis/in_hospital'
        :param predict_point:'admission/admission_after_24h/discharge'
        :return: 返回训练集文本和训练标签,记录每个样本有多少个12小时
        """
        # 加载训练集
        train_data = json.load(open(data_path + '/train_data.txt', 'r', encoding='utf-8'))
        # 加载测试疾病resArr=['desc1','desc1',...]疾病名字
        y_label = json.load(open(data_path + '/train_label.txt', 'r', encoding='utf-8'))
        new_y_label = []
        new_train_data = []
        sample_count = {}

        total = len(train_data)
        if is_sampls and sample_nums > 0:
            total = sample_nums
        # 取出每一份样本
        for i in range(total):
            every_bingli = train_data[i]
            count = 0
            if task == 'mortality':
                # 取出每一个12小时,放入new_train_data中
                for j in range(len(every_bingli)):
                    count = count + 1
                    new_data = " ".join(every_bingli[j][:-1])
                    new_train_data.append(new_data)
                # 记录每个样本有多少个12小时
                sample_count[i] = count
                # 取出每一个标签
                new_y_label.append(y_label[i][2])
            elif task == 'readmission':
                for j in range(len(every_bingli)):
                    count = count + 1
                    new_data = " ".join(every_bingli[j][:-1])
                    new_train_data.append(new_data)
                sample_count[i] = count
                new_y_label.append(y_label[i][0])
            elif task == 'in_hospital':
                for j in range(len(every_bingli)):
                    if not y_label[i][3]:
                        continue
                    count = count + 1
                    new_data = " ".join(every_bingli[j][:-1])
                    new_train_data.append(new_data)
                sample_count[i] = count
                new_y_label.append(y_label[i][3])
            elif task == 'diagnosis':
                pass
        max_sample_count = sample_count[max(sample_count)]
        print(max_sample_count)
        return new_train_data, new_y_label, sample_count

    def write_list_2_file(self, src_list, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(src_list, f)

    def save_train_data(self, train_x, train_y, sample_map, dst_path):
        self.write_list_2_file(train_x, dst_path+'_x.pkl')
        self.write_list_2_file(train_y, dst_path+'_y.pkl')
        self.write_list_2_file(sample_map, dst_path+'_map.pkl')

if __name__ == "__main__":

    data_app = Datesets()
    tasks = ['mortality', 'readmission', 'in_hospital']
    data_path = "/home/jq/PaperRealization/data"
    for task in tasks:
        texts, labels, sample_map = data_app.get_sample_datesets(data_path=data_path, task=task, sample_nums=1000, is_sampls=True)
        data_app.save_train_data(texts, labels, sample_map, '../data/'+task)

    # data_app = Datesets()
    # src_list = ['1', '2', '3']
    # data_app.write_list_2_file(src_list, '../data/demo.pkl')

    # with open('../data/in_hospital_x.pkl', 'rb')as f:
    #     test_list = pickle.load(f)
    #     print(len(test_list))


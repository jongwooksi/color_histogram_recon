import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import matplotlib.cm as cm
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, plot_confusion_matrix


dataset = '/home/iialab3/proceeding_kIIECT2023/test'

def imageProcessing(file_list, classTitle, filtersize):
    scoreList = []
    global dataset
    kernelsize = 9
    name = ""
    if classTitle == "/Abnormal_recon/" : name = "/abnormal/"
    else : name = "/normal/"

    for filename in file_list:
        print(dataset+classTitle+filename)
        img = cv2.imread(dataset+classTitle+filename)
        imgori = cv2.imread(dataset+name+filename)
     
        # dst = abs(img-imgori)
       
        # # kernel2 = np.ones((kernelsize, kernelsize), np.uint8)
        # # dst = cv2.dilate(dst, kernel2, iterations = 3)
        # # kernel3 = np.ones((kernelsize, kernelsize), np.uint8)
        # # dst = cv2.erode(dst, kernel2, iterations = 2)

        # dst[dst<200] = 0
        # #dst = dst[10:246, 10:246]

        # scoreList.append(dst.sum()/(256*256))

        # # plt.subplot(131),plt.imshow(img,cmap=cm.gray),plt.title('Original')
        # # plt.xticks([]), plt.yticks([])
        # # plt.subplot(132),plt.imshow(imgori,cmap=cm.gray),plt.title('Averaging')
        # # plt.xticks([]), plt.yticks([])
        # # plt.subplot(133),plt.imshow(dst,cmap=cm.gray),plt.title('reconstruction')
        # # plt.xticks([]), plt.yticks([])
        # # plt.show()

        #import cv2
        import matplotlib.pyplot as plt

        # 영상 불러오기
      
        # BGR을 RGB로 변환
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgori = cv2.cvtColor(imgori, cv2.COLOR_BGR2RGB)
        # 채널별 히스토그램 계산
        histogram_b = cv2.calcHist([img], [0], None, [256], [0, 256])
        histogram_g = cv2.calcHist([img], [1], None, [256], [0, 256])
        histogram_r = cv2.calcHist([img], [2], None, [256], [0, 256])

        # 그래프 그리기
        plt.figure(figsize=(8, 6))
        plt.plot(histogram_r, color='r', label='Red')
        plt.plot(histogram_g, color='g', label='Green')
        plt.plot(histogram_b, color='b', label='Blue')
        plt.xlabel('Intensity')
        plt.ylabel('Bin')
        plt.legend()
        #plt.show()

        histogram_b = cv2.calcHist([imgori], [0], None, [256], [0, 256])
        histogram_g = cv2.calcHist([imgori], [1], None, [256], [0, 256])
        histogram_r = cv2.calcHist([imgori], [2], None, [256], [0, 256])

        # 그래프 그리기
        plt.figure(figsize=(8, 6))
        plt.plot(histogram_r, color='r', label='Red')
        plt.plot(histogram_g, color='g', label='Green')
        plt.plot(histogram_b, color='b', label='Blue')
        plt.xlabel('Intensity')
        plt.ylabel('Bin')
        plt.legend()
        #plt.show()


    return scoreList

for filtersize in [7]:
    normal_file_list = os.listdir(dataset+'/Normal_recon/')
    normal_file_list.sort()

    abnormal_file_list = os.listdir(dataset+'/Abnormal_recon/')
    abnormal_file_list.sort()


    AbnormalList = imageProcessing(abnormal_file_list, "/Abnormal_recon/",filtersize)

    NormalList = imageProcessing(normal_file_list, "/Normal_recon/",filtersize)
    AbnormalList.sort(reverse=True)
    NormalList.sort(reverse=True)
    print(AbnormalList)
    print(NormalList)

    print(sum(AbnormalList[:50])/50)
    print(sum(NormalList[:50])/50)
    
    # x_data = NormalList+AbnormalList

    # x_data = (x_data - min(x_data)) / (max(x_data) - min(x_data) + 0.000001)

    # normal_true = [0 for i in range(len(normal_file_list))]
    # abnormal_true = [1 for i in range(len(abnormal_file_list))]
    
    # fpr, tpr, threshold = roc_curve(normal_true+abnormal_true, x_data)
    # optimal_index = np.argmax(tpr - fpr) 
    # optimal_threshold = threshold[optimal_index]
    # print(optimal_threshold)
    # sumTrue = normal_true + abnormal_true
    # sumPred= [0 if i<optimal_threshold else 1 for i in x_data]

    # cnf_matrix = confusion_matrix(sumTrue, sumPred)

    # print("AUC= %0.3f"%( roc_auc_score(sumTrue, sumPred)))
    # print(classification_report(sumTrue, sumPred,digits=3))
    # print(cnf_matrix)

    
      
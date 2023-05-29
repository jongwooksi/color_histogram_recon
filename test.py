import cv2
import numpy as np
import os
import tensorflow as tf
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, plot_confusion_matrix


def imread(filename, flags=cv2.IMREAD_COLOR, dtype=np.uint8): 
    try: 
        n = np.fromfile(filename, dtype) 
        img = cv2.imdecode(n, flags) 
        return img 
    except Exception as e: 
        print(e) 
        return None

def imwrite(filename, img, params=None): 
    try: 
        ext = os.path.splitext(filename)[1] 
        result, n = cv2.imencode(ext, img, params) 
        if result: 
            with open(filename, mode='w+b') as f: 
                n.tofile(f) 
            return True 
        else: 
            return False 
    except Exception as e: 
        print(e) 
        return False


def datapreprocessing(folderlist, PATH, dtype):
    x_test = []
    tmplist= [] 
    cnt = 1000
    aa = []
    '''
    for folder in folderlist:
        if dtype == "Normal_input" :
            file_list = os.listdir(PATH + folder)
            file_list.sort()
            for filename in file_list:
                if folder != "good" : continue 
                filepath = PATH + folder+'/' + filename
                image = imread(filepath)
                print(folder)
                image = cv2.resize(image, (256,256))
                cv2.imwrite("Test/normal/"+str(cnt)+".png",image)
                image.tolist()
                image = image / 255.
                x_test.append(image)
                tmplist.append(filepath)
                cnt += 1
                aa.append(filename)
        else:
            file_list = os.listdir(PATH + folder)
            file_list.sort()
            for filename in file_list:
                if folder == "good" : continue

                filepath = PATH + folder + '/' + filename
                image = imread(filepath)
                print(folder)
                image = cv2.resize(image, (256,256))
                cv2.imwrite("Test/abnormal/"+str(cnt)+".png",image)
                image.tolist()
                image = image / 255.
                x_test.append(image)
                tmplist.append(filepath)
                cnt += 1
                aa.append(filename)
    '''       
        
    
    if dtype == "Normal_input" :
        file_list = os.listdir('./test/normal/')
        file_list.sort()
        for filename in file_list:
            filepath = './test/normal/' + filename
            image = imread(filepath)
            print(filepath)

            image.tolist()
            image = image / 255.
            x_test.append(image)
            tmplist.append(filepath)
            cnt += 1
            aa.append(filename)
    else:
       
        file_list = os.listdir('./test/abnormal/')
        file_list.sort()
        for filename in file_list:
            filepath = './test/abnormal/' + filename
            image = imread(filepath)
            print(filepath)
            #image = cv2.resize(image, (256,256))
            #cv2.imwrite("Test/abnormal/"+str(cnt)+".png",image)
            image.tolist()
            image = image / 255.
            x_test.append(image)
            tmplist.append(filepath)
            cnt += 1
            aa.append(filename)
    return x_test, aa, tmplist

def fft(img):
    img = np.mean(img,axis=2)

    rows, cols = img.shape 
    crow,ccol = round(rows/2), round(cols/2) 

    dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    mask = np.zeros((rows,cols,2),np.uint8)
    mask[crow-50:crow+50,ccol-50:ccol+50] = 1 

    fshift = dft_shift*mask 
    f_ishift = np.fft.ifftshift(fshift) 
    img_back = cv2.idft(f_ishift) 
    img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])

    return img_back

def fftreverse(img):
    img = np.mean(img,axis=2)

    rows, cols = img.shape 
    crow,ccol = round(rows/2), round(cols/2) 

    dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    mask = np.ones((rows,cols,2),np.uint8)
    mask[crow-10:crow+10,ccol-10:ccol+10] = 0 

    fshift = dft_shift*mask 
    f_ishift = np.fft.ifftshift(fshift) 
    img_back = cv2.idft(f_ishift) 
    img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])

    return img_back

def calCropScore( x, pred, *file):
    #fftx = fftreverse(x/255)/255.0
    fftpred = fftreverse(pred)/255.0

    #absval2 = abs(fftx-fftpred)
    #x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    #pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
    # fig = plt.figure()
    # ax1 = fig.add_subplot(1, 3, 1)
    # ax1.imshow(x)
    # ax1.set_title('dst1')
    # ax1.axis("off")
    # ax2 = fig.add_subplot(1, 3, 2)
    # ax2.imshow(pred)
    # ax2.set_title('dst2')
    # ax2.axis("off")
    # ax3 = fig.add_subplot(1, 3, 3)
    # ax3.imshow(fftpred*255, cmap='gray')
    # ax3.set_title('dst3')
    # ax3.axis("off")
    # plt.show()

    absval2 = np.array(fftpred).flatten()
    absval2 = sorted(absval2, reverse = True)

    score2 = np.mean(absval2)

    return score2

def calScore(x_test, prediction, classv,name):
    imageShape = (256, 256,3)
    tempList = []
  
    for i in range(0,x_test.shape[0]):
        reconScore = calCropScore(x_test[i], prediction[i],classv, name[i])
      
        tempList.append(reconScore)
        
        if classv == "normal":
            cv2.imwrite("./test/Normal_recon/"+name[i], prediction[i]*255)
        else:
           
            cv2.imwrite("./test/Abnormal_recon/"+name[i],prediction[i]*255)
        
    return tempList
    

def unetDeep():
    input_shape=(256,256,3)
    n_channels = input_shape[-1]
    inputs = tf.keras.layers.Input(input_shape)

    conv1 = tf.keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = tf.keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    drop1 = tf.keras.layers.Dropout(0.2)(conv1)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(drop1)

    conv2 = tf.keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = tf.keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    drop2 = tf.keras.layers.Dropout(0.2)(conv2)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(drop2)

    conv3 = tf.keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = tf.keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    drop3 = tf.keras.layers.Dropout(0.2)(conv3)
    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(drop3)
   
    conv4 = tf.keras.layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = tf.keras.layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = tf.keras.layers.Dropout(0.2)(conv4)
    
    pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(drop4)
    conv5 = tf.keras.layers.Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = tf.keras.layers.Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='conv5')(conv5)
    drop5 = tf.keras.layers.Dropout(0.2)(conv5)

    up6 = tf.keras.layers.Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(tf.keras.layers.UpSampling2D(size = (2,2))(drop5))
    merge6 = tf.keras.layers.concatenate([drop4,up6], axis = 3)
    conv6 = tf.keras.layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    #conv6 = tf.keras.layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(up6)
    drop6 = tf.keras.layers.Dropout(0.2)(conv6)

    up7 = tf.keras.layers.Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(tf.keras.layers.UpSampling2D(size = (2,2))(drop6))
    merge7 = tf.keras.layers.concatenate([drop3,up7], axis = 3)
    conv7 = tf.keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    #conv7 = tf.keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(up7)
    drop7 = tf.keras.layers.Dropout(0.2)(conv7)

    up8 = tf.keras.layers.Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(tf.keras.layers.UpSampling2D(size = (2,2))(drop7))
    merge8 = tf.keras.layers.concatenate([drop2,up8], axis = 3)
    conv8 = tf.keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    #conv8 = tf.keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(up8)
    drop8 = tf.keras.layers.Dropout(0.2)(conv8)

    up9 = tf.keras.layers.Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(tf.keras.layers.UpSampling2D(size = (2,2))(drop8))
    merge9 = tf.keras.layers.concatenate([drop1,up9], axis = 3)
    conv9 = tf.keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    #conv9 = tf.keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(up9)
    conv9 = tf.keras.layers.Conv2D(n_channels, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
 
    model = tf.keras.Model(inputs, conv9)

    return model


if __name__ == "__main__" :
    
    NORMAL_PATH = './test/'
    ABNORMAL_PATH = './test/'
    
    folder_list = os.listdir(NORMAL_PATH)
    folder_list.sort()

    abfolder_list = os.listdir(ABNORMAL_PATH)
    abfolder_list.sort()

    x_test_normal, normalfile_list, _ = datapreprocessing(folder_list, NORMAL_PATH, "Normal_input")
    x_test_abnormal, abnormalfile_list, tmplist = datapreprocessing(abfolder_list, ABNORMAL_PATH,"Abnormal_input")
    
    x_test_normal = np.array(x_test_normal)
    x_test_abnormal = np.array(x_test_abnormal)

    model = unetDeep()
    model.load_weights("./model/data-0100.ckpt")
    ##model = tf.keras.models.load_model('solarModel2')
    
    predictions_normal = model.predict(x_test_normal)
    predictions_abnormal = model.predict(x_test_abnormal)
    NormalList = calScore(x_test_normal, predictions_normal,"normal",normalfile_list)
    AbnormalList = calScore(x_test_abnormal, predictions_abnormal,"abnormal",abnormalfile_list)

    x_data = NormalList+AbnormalList
    x_data = (x_data - min(x_data)) / (max(x_data) - min(x_data))

    normal_true = [0 for i in range(int(x_test_normal.shape[0]))]
    abnormal_true = [1 for i in range(int(x_test_abnormal.shape[0]))]
    print(normal_true, abnormal_true)
    fpr, tpr, threshold = roc_curve(normal_true+abnormal_true, x_data)
    optimal_index = np.argmax(tpr - fpr) 
    optimal_threshold = threshold[optimal_index]
    
    sumTrue = normal_true + abnormal_true
    sumPred= [0 if i<optimal_threshold else 1 for i in x_data]

    print(x_data[:int(x_test_normal.shape[0])], sumPred[:int(x_test_normal.shape[0])])
    print(x_data[int(x_test_normal.shape[0]):], sumPred[int(x_test_normal.shape[0]):])
     
    cnf_matrix = confusion_matrix(sumTrue, sumPred)
  
    print(classification_report(sumTrue, sumPred,digits=3))
    print(cnf_matrix)
    print(optimal_threshold)
    
    plt.plot(fpr, tpr, linewidth=2,label="ROC curve (area = %0.3f)" % roc_auc_score(sumTrue, sumPred))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend(loc="lower right")   
    plt.show() 

    plt.hist(x_data[:int(x_test_normal.shape[0])], alpha=0.7, bins=100, label='Normal')
    plt.hist(x_data[int(x_test_normal.shape[0]):], alpha=0.7, bins=100, label='Defect')
    plt.legend(loc='upper right')
    plt.show()
    
    print(sum(x_data[:int(x_test_normal.shape[0])])/int(x_test_normal.shape[0]))
    print(sum(x_data[int(x_test_normal.shape[0]):])/int(x_test_abnormal.shape[0]))
    
    # for i in range(x_test_abnormal.shape[0]):
    #     if sumPred[i+x_test_normal.shape[0]] == 0:
    #         print(tmplist[i], sumPred[i+x_test_normal.shape[0]])
  
    
    # for i in range(x_test_normal.shape[0]):
    #     print(normalfile_list[i], sumPred[i])

    
    # for i in range(x_test_abnormal.shape[0]):
    #     print(tmplist[i], sumPred[i+x_test_normal.shape[0]])
    
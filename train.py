import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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
def get_gaussian_kernel(size, sigma):
    """가우시안 필터 생성 함수"""
    coords = tf.range(-size // 2 + 1, size // 2 + 1)
    coords = tf.cast(coords, tf.float32)
    coords = tf.expand_dims(coords, axis=0)
    coords = tf.expand_dims(coords, axis=-1)
    coords = tf.tile(coords, [1, size, 1]) + tf.transpose(coords, perm=[0, 2, 1])
    coords = tf.expand_dims(coords, axis=-1)

    gaussian_kernel = tf.math.exp(-coords / (2 * sigma * sigma))
    gaussian_kernel /= tf.reduce_sum(gaussian_kernel)

    return gaussian_kernel

def ssim_loss(y_true, y_pred):
    K1 = 0.01
    K2 = 0.03
    L = 1  # dynamic range of pixel values in the image (e.g., 255 for 8-bit grayscale images)

    # Convert y_true and y_pred to grayscale
    y_true_gray = tf.image.rgb_to_grayscale(y_true)
    y_pred_gray = tf.image.rgb_to_grayscale(y_pred)

    # Compute mean, variance, and covariance of y_true_gray and y_pred_gray
    mu_true = tf.nn.conv2d(y_true_gray, tf.ones([11, 11, 1, 1]) / 121, strides=[1, 1, 1, 1], padding='VALID')
    mu_pred = tf.nn.conv2d(y_pred_gray, tf.ones([11, 11, 1, 1]) / 121, strides=[1, 1, 1, 1], padding='VALID')
    sigma_true_squared = tf.nn.conv2d(y_true_gray ** 2, tf.ones([11, 11, 1, 1]) / 121, strides=[1, 1, 1, 1], padding='VALID') - mu_true ** 2
    sigma_pred_squared = tf.nn.conv2d(y_pred_gray ** 2, tf.ones([11, 11, 1, 1]) / 121, strides=[1, 1, 1, 1], padding='VALID') - mu_pred ** 2
    sigma_true_pred = tf.nn.conv2d(y_true_gray * y_pred_gray, tf.ones([11, 11, 1, 1]) / 121, strides=[1, 1, 1, 1], padding='VALID') - mu_true * mu_pred

    # Compute SSIM
    numerator = (2 * mu_true * mu_pred + K1) * (2 * sigma_true_pred + K2)
    denominator = (mu_true ** 2 + mu_pred ** 2 + K1) * (sigma_true_squared + sigma_pred_squared + K2)
    ssim = numerator / denominator

    # Return 1 - SSIM as the loss (to minimize)
    return 1 - tf.reduce_mean(ssim)

def psnr(y_true, y_pred, max_val=255.0):
    """PSNR (Peak Signal-to-Noise Ratio) 손실 함수

    Args:
        y_true (tf.Tensor): 정답 이미지 (batch_size, height, width, channels)
        y_pred (tf.Tensor): 예측 이미지 (batch_size, height, width, channels)
        max_val (float, optional): 이미지의 최대값. 기본값은 255.0.

    Returns:
        tf.Tensor: PSNR 손실 값
    """
    mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=[1, 2, 3])
    psnr = 10.0 * tf.math.log(max_val**2 / mse) / tf.math.log(10.0)
    return tf.reduce_mean(psnr)

def image_distribution_loss(y_true, y_pred, num_bins=256):
    """RGB 3채널 이미지 전체에 대한 분포를 학습하기 위한 Loss 함수"""
   
    # 입력 이미지와 출력 이미지를 [0, 1] 범위로 정규화
    #y_true = tf.cast(y_true, dtype=tf.float32) / 255.0
    #y_pred = tf.cast(y_pred, dtype=tf.float32) / 255.0
    # y_true = tf.tile(y_true, [1, 1, 1, 3])
    # y_pred = tf.tile(y_pred, [1, 1, 1, 3])
    y_true = tf.cast(y_true, tf.float32)/255.0
    y_pred = tf.cast(y_pred, tf.float32)/255.0
    # 히스토그램 계산
    true_histograms = [
        tf.histogram_fixed_width(tf.cast(y_true, tf.float32), value_range=(0.0, 1.0), nbins=num_bins, dtype=tf.int32)
        for i in range(3)
    ]
    pred_histograms = [
        tf.histogram_fixed_width(tf.cast(y_pred, tf.float32), value_range=(0.0, 1.0), nbins=num_bins, dtype=tf.int32)
        for i in range(3)
    ]

    # 각 채널 별로 히스토그램 차이를 계산하여 Loss 값 생성
    channel_losses = [
        tf.reduce_mean(tf.abs(tf.cast(true_histograms, tf.float32) - tf.cast(pred_histograms, tf.float32)))
        for i in range(3)
    ]

    # 채널 별 Loss 값들의 평균을 계산하여 최종 Loss 값 얻음
    loss = tf.reduce_mean(channel_losses)

    # 손실 계산
    #loss = tf.reduce_mean(tf.abs(tf.cast(hist_true, tf.float32) - tf.cast(hist_pred, tf.float32)))

    return loss
def image_sharpening_loss(y_true, y_pred, weight=0.5):
    """이미지 샤프닝을 위한 Loss 함수"""

    # 입력 이미지와 출력 이미지의 채널 수를 동일하게 맞춤
    # 입력 이미지와 출력 이미지의 차원을 3배로 확장
    y_true = tf.tile(y_true, [1, 1, 1, 3])
    y_pred = tf.tile(y_pred, [1, 1, 1, 3])

    # Sobel 필터를 사용하여 입력 이미지와 출력 이미지의 경계 성분을 계산
    kernel = tf.constant([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])
    kernel = tf.reshape(kernel, [3, 3, 1, 1])
    gx_true = tf.nn.conv2d(y_true, kernel, strides=[1, 1, 1, 1], padding='SAME')
    gx_pred = tf.nn.conv2d(y_pred, kernel, strides=[1, 1, 1, 1], padding='SAME')
    gy_true = tf.nn.conv2d(y_true, kernel, strides=[1, 1, 1, 1], padding='SAME')
    gy_pred = tf.nn.conv2d(y_pred, kernel, strides=[1, 1, 1, 1], padding='SAME')

    # 경계 성분의 차이를 계산하여 Loss 값 생성
    gx_diff = tf.abs(gx_true - gx_pred)
    gy_diff = tf.abs(gy_true - gy_pred)
    loss = tf.reduce_mean(gx_diff + gy_diff) * weight

    return loss
def myloss(true, prediction):
   
    L1loss = tf.abs(true - prediction)
    ssimloss = ssim_loss(true, prediction)
    return L1loss+ image_distribution_loss(true, prediction)
    
if __name__ == "__main__" :

    TRAIN_PATH = './train/good'

    x_train = []
    y_train = []

    file_list = os.listdir(TRAIN_PATH)
    file_list.sort()
    for filename in file_list:
        filepath = TRAIN_PATH +  '/' + filename
        print(filename)
        image = imread(filepath)
        image.tolist()
        image = image / 255.
        x_train.append(image)
    
    train_datagen = ImageDataGenerator(
  
        shear_range=0.2,
        zoom_range=0.2,
        vertical_flip=True,
        horizontal_flip=True)
    
    x_train = np.array(x_train)

    x_train = x_train.reshape(-1,256,256,3)

    model = unetDeep()
    model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(lr = 1e-4), loss=myloss)

    
    checkpoint_path = "model/data-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)


    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, 
        verbose=1, 
        save_weights_only=True,
        period=10)


    model.fit_generator(train_datagen.flow(x_train, x_train, batch_size=16), epochs=100,callbacks=[cp_callback])
    
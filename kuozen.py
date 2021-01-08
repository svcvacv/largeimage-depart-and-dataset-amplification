#coding=utf-8
import skimage
import io,os
import matplotlib.pyplot as plt
from PIL import Image,ImageEnhance
from PIL import ImageChops
import cv2
from skimage import io
import numpy as np
import random
def move(root_path,img_name,off): #平移，平移尺度为off
    img = Image.open(os.path.join(root_path, img_name))
    offsets=ImageChops.offset(img,off,0)
    return offsets

def flip(root_path,img_name):   #翻转图像
    img = Image.open(os.path.join(root_path, img_name))
    filp_img = img.transpose(Image.FLIP_LEFT_RIGHT)
    # filp_img.save(os.path.join(root_path,img_name.split('.')[0] + '_flip.jpg'))
    return filp_img

def aj_contrast(root_path,img_name): #调整对比度 两种方式 gamma/log
    image = skimage.io.imread(os.path.join(root_path, img_name))
    gam= skimage.exposure.adjust_gamma(image, 0.5)
    # skimage.io.imsave(os.path.join(root_path,img_name.split('.')[0] + '_gam.jpg'),gam)
    log= skimage.exposure.adjust_log(image)
    # skimage.io.imsave(os.path.join(root_path,img_name.split('.')[0] + '_log.jpg'),log)
    return gam,log
def rotation(root_path, img_name):
    img = Image.open(os.path.join(root_path, img_name))
    rotation_img = img.rotate(80) #旋转角度
    # rotation_img.save(os.path.join(root_path,img_name.split('.')[0] + '_rotation.jpg'))
    return rotation_img

def rotation2(root_path, img_name):
    img = Image.open(os.path.join(root_path, img_name))
    rotation_img = img.rotate(150) #旋转角度
    # rotation_img.save(os.path.join(root_path,img_name.split('.')[0] + '_rotation.jpg'))
    return rotation_img

def rotation3(root_path, img_name):
    img = Image.open(os.path.join(root_path, img_name))
    rotation_img = img.rotate(240) #旋转角度
    # rotation_img.save(os.path.join(root_path,img_name.split('.')[0] + '_rotation.jpg'))
    return rotation_img

def randomGaussian(root_path, img_name, mean, sigma):  #高斯噪声
    image = Image.open(os.path.join(root_path, img_name))
    im = np.array(image)
    #设定高斯函数的偏移
    means = 0
    #设定高斯函数的标准差
    sigma = 25
    #r通道
    r = im[:,:,0].flatten()

    #g通道
    g = im[:,:,1].flatten()

    #b通道
    b = im[:,:,2].flatten()

    #计算新的像素值
    for i in range(im.shape[0]*im.shape[1]):

        pr = int(r[i]) + random.gauss(0,sigma)

        pg = int(g[i]) + random.gauss(0,sigma)

        pb = int(b[i]) + random.gauss(0,sigma)

        if(pr < 0):
            pr = 0
        if(pr > 255):
            pr = 255
        if(pg < 0):
            pg = 0
        if(pg > 255):
            pg = 255
        if(pb < 0):
            pb = 0
        if(pb > 255):
            pb = 255
        r[i] = pr
        g[i] = pg
        b[i] = pb
    im[:,:,0] = r.reshape([im.shape[0],im.shape[1]])

    im[:,:,1] = g.reshape([im.shape[0],im.shape[1]])

    im[:,:,2] = b.reshape([im.shape[0],im.shape[1]])
    gaussian_image = gaussian_image = Image.fromarray(np.uint8(im))
    return gaussian_image
def randomColor(root_path, img_name): #随机颜色
    """
    对图像进行颜色抖动
    :param image: PIL的图像image
    :return: 有颜色色差的图像image
    """
    image = Image.open(os.path.join(root_path, img_name))
    random_factor = np.random.randint(10, 21) / 10.  # 随机因子
    color_image = ImageEnhance.Color(image).enhance(random_factor)  # 调整图像的饱和度
    random_factor = np.random.randint(12, 19) / 10.  # 随机因子
    brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  # 调整图像的亮度
    random_factor = np.random.randint(12, 29) / 10.  # 随机因子
    contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  # 调整图像对比度
    random_factor = np.random.randint(10, 21) / 10.  # 随机因子
    return ImageEnhance.Sharpness(contrast_image).enhance(random_factor)  # 调整图像锐度


def random_crop(root_path, img_name ):  # 随机裁剪
    min_ratio=0.6
    max_ratio=1.0
    image = cv2.imread(root_path + "/" + img_name)

    h, w = image.shape[:2]
    ratio = random.random()
    scale = min_ratio + ratio * (max_ratio - min_ratio)
    new_h = int(h*scale)    
    new_w = int(w*scale)
    y = np.random.randint(0, h - new_h)    
    x = np.random.randint(0, w - new_w)
    image = image[y:y+new_h, x:x+new_w, :]
    return image


def mkdir(path):
    # 引入模块
    import os
 
    # 去除首位空格
    path=path.strip()
    # 去除尾部 \ 符号
    path=path.rstrip("\\")
 
    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists=os.path.exists(path)
 
    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path) 
 
        print(path+' 创建成功')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print(path+' 目录已存在')
        return False


array_of_img = [] # this if for store all of the image data
# this function is for read image,the input is directory name



def read_directory(directory_name):
    i=1
    # this loop is for read each image in this foder,directory_name is the foder name with images
    path_list = os.listdir(r"./"+directory_name)
    path_list.sort(key=lambda x:int(x.split('.')[0]))
    for filename in path_list:
        j=1
        name=str(i) 
        i=i+1
        mkdir("./"+name)

        for num1 in range(1,14):
            for num2 in range(1,15):
                if (num1 == 1):
                    imgname=str(j)   
                    imgs=randomGaussian(r"./"+directory_name, filename,0,18)
                    imgs.save("./"+name+"/"+imgname+".jpg")
                    j=j+1
                if (num1 == 2):
                    imgname=str(j)   
                    imgs=randomColor(r"./"+directory_name, filename)
                    imgs.save("./"+name+"/"+imgname+".jpg")
                    j=j+1
                if (num1 == 3):
                    imgname=str(j)   
                    imgs=rotation(r"./"+directory_name, filename)
                    imgs.save("./"+name+"/"+imgname+".jpg")
                    j=j+1
                if (num1 == 4):
                    imgname=str(j)   
                    imgsg,imgsl=aj_contrast(r"./"+directory_name, filename)
                    skimage.io.imsave("./"+name+"/"+imgname+".jpg", imgsg)
                    j=j+1
                if (num1 == 5):
                    imgname=str(j)   
                    imgsg,imgsl=aj_contrast(r"./"+directory_name, filename)
                    skimage.io.imsave("./"+name+"/"+imgname+".jpg", imgsl)
                    j=j+1
                if (num1 == 6):
                    imgname=str(j)   
                    imgs=flip(r"./"+directory_name, filename)
                    imgs.save("./"+name+"/"+imgname+".jpg")
                    j=j+1
                if (num1 == 7):
                    imgname=str(j)   
                    imgs=move(r"./"+directory_name, filename,10)
                    imgs.save("./"+name+"/"+imgname+".jpg")
                    j=j+1
                if (num1 == 8):
                    imgname=str(j)   
                    imgs=rotation2(r"./"+directory_name, filename)
                    imgs.save("./"+name+"/"+imgname+".jpg")
                    j=j+1
                if (num1 == 9):
                    imgname=str(j)   
                    imgs=rotation3(r"./"+directory_name, filename)
                    imgs.save("./"+name+"/"+imgname+".jpg")
                    j=j+1
                if (num1 == 10):
                    imgname=str(j)   
                    imgs=randomColor(r"./"+directory_name, filename)
                    imgs.save("./"+name+"/"+imgname+".jpg")
                    j=j+1
                if (num1 == 11):
                    imgname=str(j)   
                    imgs=random_crop(r"./"+directory_name, filename)
                    cv2.imwrite("./"+name+"/"+imgname+".jpg",imgs)
                    j=j+1
                if (num1 == 12):
                    imgname=str(j)   
                    imgs=random_crop(r"./"+directory_name, filename)
                    cv2.imwrite("./"+name+"/"+imgname+".jpg",imgs)
                    j=j+1
                if (num1 == 13):
                    imgname=str(j)   
                    imgs=random_crop(r"./"+directory_name, filename)
                    cv2.imwrite("./"+name+"/"+imgname+".jpg",imgs)
                    j=j+1


                if (num2 == 1):
                    imgs=randomGaussian("./"+name+"/", imgname+".jpg",0,18)
                    imgs.save("./"+name+"/"+imgname+".jpg")
                if (num2 == 2):
                    imgs=randomColor("./"+name+"/", imgname+".jpg")
                    imgs.save("./"+name+"/"+imgname+".jpg")
                if (num2 == 3):
                    imgs=rotation("./"+name+"/", imgname+".jpg")
                    imgs.save("./"+name+"/"+imgname+".jpg")
                if (num2 == 4):
                    imgsg,imgsl=aj_contrast("./"+name+"/", imgname+".jpg")
                    skimage.io.imsave("./"+name+"/"+imgname+".jpg",imgsg)
                if (num2 == 5):
                    imgsg,imgsl=aj_contrast("./"+name+"/", imgname+".jpg")
                    skimage.io.imsave("./"+name+"/"+imgname+".jpg",imgsl)
                if (num2 == 6):
                    imgs=flip("./"+name+"/", imgname+".jpg")
                    imgs.save("./"+name+"/"+imgname+".jpg")
                if (num2 == 7):
                    imgs=move("./"+name+"/", imgname+".jpg",10)
                    imgs.save("./"+name+"/"+imgname+".jpg")
                if (num2 == 8):
                    j=j
                if (num2 == 9):
                    imgs=rotation2("./"+name+"/", imgname+".jpg")
                    imgs.save("./"+name+"/"+imgname+".jpg")
                if (num2 == 10):
                    imgs=rotation3("./"+name+"/", imgname+".jpg")
                    imgs.save("./"+name+"/"+imgname+".jpg")
                if (num2 == 11):
                    imgs=randomColor("./"+name+"/", imgname+".jpg")
                    imgs.save("./"+name+"/"+imgname+".jpg")
                if (num2 == 12):
                    imgs=random_crop("./"+name+"/", imgname+".jpg")
                    cv2.imwrite("./"+name+"/"+imgname+".jpg",imgs)
                if (num2 == 13):
                    imgs=random_crop("./"+name+"/", imgname+".jpg")
                    cv2.imwrite("./"+name+"/"+imgname+".jpg",imgs)
                if (num2 == 14):
                    imgs=random_crop("./"+name+"/", imgname+".jpg")
                    cv2.imwrite("./"+name+"/"+imgname+".jpg",imgs)

root_path="../jpgnew"
read_directory("../jpgnew")#这里传入所要读取文件夹的绝对路径，加引号（引号不能省略！）








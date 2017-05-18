# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
def get_info_from_image(path):

    import numpy as np
    
    import pickle
    from skimage import io

    from skimage.color import rgb2gray
    from skimage.transform import downscale_local_mean
    from sklearn.decomposition import PCA
    from sklearn.svm import LinearSVC
    
    
    loadfile = open('models_hack.obj','r')
    pca,classif=pickle.load(loadfile)  
    loadfile.close()
    image = io.imread(path)
    img_gray = rgb2gray(image)
    red_img_gray=downscale_local_mean(img_gray, (20, 20))
    vtemp=red_img_gray.flatten()

        
#    pca = PCA(n_components=200)
#have to Load pca
    x=pca.transform(vtemp);

#Have to load classif
#classif = (LinearSVC(verbose=1,C=0.01))

    pred=classif.predict(x);
                        
    return pred                    



pred=get_info_from_image('IMG_8768_1.JPG')
print(pred)


import numpy as np
import cv2
from matplotlib import pyplot as plt


scores = []


def match(img1, img2):
    res = cv2.matchTemplate(img1, img2, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
    p = img1 / (np.max(img1))
    t = img2 / (np.max(img2))
    
    p = p.flatten()
    t = t.flatten()
    
    
    p = (p - np.mean(p)) / (len(p)*np.std(p))
    t = (t - np.mean(t)) / (np.std(t))
    c = np.sum(np.dot(p,t))
    return max_val,c

def metrics(p,t, smooth=1,alpha=.1,beta=.9):
    p = p / (np.max(p))
    t = t / (np.max(t))
    
    p = p.flatten()
    t = t.flatten()
    
    
    tp = np.sum(np.dot(p,t))
    fp = np.sum(np.dot(p, (1-t)))
    fn = np.sum(np.dot((1-p),t))
    
    # scores
    dice = (2 * tp) / (2 * tp + fp + fn)
    iou = tp / (tp + fp + fn)
    sensitivity = tp / (tp + fn)
    precision = tp / (tp + fp)
    f2 = (5 * tp ) / (4 * (tp + fn) + fn + fp)
    tversky = (tp + smooth) / (tp + alpha*fp + beta*fn + smooth)  

    # scores.append([dice, iou,sensitivity, precision,f2])
    
    
    return dice, iou, sensitivity, precision, f2, tversky


def Reverse(lst):
    new_lst = lst[::-1]
    return new_lst



shifting_parameters = np.arange(256,448,20, dtype="int")
for i in Reverse(shifting_parameters):    
    img = np.zeros((448,448), np.uint8)
    img_ = np.zeros((448,448), np.uint8)
    
    img1 = cv2.circle(img, (224,224),25,(255,255,255), 150)
    img2 = cv2.circle(img_, (i,224),25,(255,255,255), 150)
    
    
    dice, iou, sensitivity, precision, f2, tversky = metrics(img1, img2)
    ncc, ncc1 = match(img1, img2)
    scores.append([ncc1,tversky,iou])

ncc = [row[0] for row in scores]
tversky = [row[1] for row in scores]
iou = [row[2] for row in scores]




# ncc = Reverse(ncc)
# tversky = Reverse(tversky)
# iou = Reverse(iou)


fig, ax = plt.subplots()
plt.rcParams["font.family"] = "Times New Roman"
plt.plot(range(0, 1))
plt.xlabel("shifting parameters in pixels", fontsize = 20)
plt.ylabel("matching indexes", fontsize = 20)
plt.xlim(0,180,10)
plt.ylim(-.2,1)
x = np.linspace(0,180,10, dtype="int")
x.reshape((1,10))
# plt.scatter(ncc,tversky,iou)
plt.plot(x, ncc, '-r', label='NCC', marker="o")
plt.plot(x, tversky, '-g', label='TI', marker = "o")
plt.plot(x, iou, '--k', label='IoU', marker ="o")

legend = ax.legend(loc='upper center', shadow=False, fontsize='x-large')
#legend.get_frame().set_facecolor('C0')

#leg = ax.legend();
plt.savefig("iou-ncc-tversky-loss.png", dpi=600)
#plt.show()






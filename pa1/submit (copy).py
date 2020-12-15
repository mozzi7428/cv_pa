import numpy as np
import cv2
import os
import glob
import gco
import matplotlib.pyplot as plt


def image_alignment(template, image, sift_thres=0.6, ransac_thres=1.0):
    # convert to gray
    tmp = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # compute keypoints and descriptors
    sift = cv2.xfeatures2d.SIFT_create()
    img_kp, img_dsc = sift.detectAndCompute(img, None)
    tmp_kp, tmp_dsc = sift.detectAndCompute(tmp, None)

    # find where are good matches
    matcher = cv2.BFMatcher_create(normType=cv2.NORM_L2, crossCheck=False)
    matches = matcher.knnMatch(img_dsc, tmp_dsc, k=2)
    good_matches = []
    for m1, m2 in matches:
        if m1.distance < sift_thres * m2.distance:
            good_matches.append(m1)

    # apply homography
    img_pts = np.float32([img_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    tmp_pts = np.float32([tmp_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    homo, mask = cv2.findHomography(img_pts, tmp_pts, cv2.RANSAC, ransac_thres)

    h, w, _ = template.shape
    image_aligned = cv2.warpPerspective(image, homo, (w, h))
    return image_aligned




def align(images_path, dataset, results_path):
    if not os.path.isdir(os.path.join(results_path, dataset)):
        os.makedirs(os.path.join(results_path, dataset))

    image_files = sorted(glob.glob(os.path.join(images_path, dataset, '*')))
    i=2
    template = cv2.imread(image_files[1])
    image = cv2.imread(image_files[2])
    aligned = image_alignment(template, image)

    error = []
    error.append(np.abs(template - image))
    error.append(np.abs(template - aligned))
    print(np.mean(error[0]), np.mean(error[1]))
    error = np.concatenate(error, axis=1)
    print(np.min(error), np.max(error))


    cv2.imwrite(os.path.join(results_path, 'errormap_%s.png'%dataset), error)




def get_focus(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = cv2.GaussianBlur(img, (3,3), 0)
    img_x = cv2.Sobel(img,cv2.CV_32F,1,0,ksize=3)
    img_y = cv2.Sobel(img,cv2.CV_32F,0,1,ksize=3)
    tenenbaum = img_x*img_x + img_y*img_y
    return tenenbaum

def depth_focus(aligned_path, dataset, results_path):
    if not os.path.isdir(os.path.join(results_path, dataset)):
        os.makedirs(os.path.join(results_path, dataset))
    image_files = sorted(glob.glob(os.path.join(aligned_path, dataset, '*')))

    imgs = []
    focus = []
    # get focus maps
    for f in image_files:
        img = cv2.imread(f)
        img_focus = get_focus(img)
        imgs.append(img.astype(np.float32)/255.0)
        focus.append(img_focus)

    # initial depth map
    focus = np.array(focus)
    depth_map = np.argmax(focus, axis=0)

    plt.imshow(depth_map)
    plt.title('depthmap_%s' % (dataset))
    plt.colorbar()
    plt.savefig(os.path.join(results_path, dataset, 'depthmap.png'))
    # plt.show()

    # retrieve focused image
    focused = np.zeros(img.shape)
    for i in range(depth_map.shape[0]):
        for j in range(depth_map.shape[1]):
            focused[i,j,:] = imgs[depth_map[i, j]][i,j,:]
    cv2.imwrite(os.path.join(results_path, dataset, '01focused.png' ), (focused*255))

    np.save(os.path.join(results_path, dataset, 'focus.npy'), focus)


def graph_cut(aligned_path, dataset, results_path):
    # load images
    image_files = sorted(glob.glob(os.path.join(images_path, dataset, '*')))
    imgs = [cv2.imread(f) for f in image_files]

    # load and scale focusmaep
    focus = np.load(os.path.join(results_path, dataset, 'focus.npy'))

    # unary cost
    unary = np.transpose(focus, [1, 2, 0]) / np.max(focus)
    unary = (1 - unary) * (focus.shape[0]-1)
    unary = np.ascontiguousarray(unary)

    # pairwise cost
    l = focus.shape[0]
    pairwise = np.zeros((l,l))
    for i in range(l):
        for j in range(l):
            pairwise[i,j] = abs(i-j)
    pairwise = pairwise.astype(np.float32)
    pairwise = np.ascontiguousarray(pairwise)

    # vertial/horizontal cost
    sigma = 1.1
    edges = np.argmax(focus, axis = 0) / np.max(focus)
    vertical = np.exp(-abs(edges[:-1, :] - edges[1:, :]) / (2*sigma**2))
    horizontal = np.exp(-abs(edges[:, :-1] - edges[:, 1:]) / (2*sigma**2))

    # conduct graph cut
    print('graph cut')
    focusmap = gco.cut_grid_graph(unary_cost=unary, pairwise_cost=pairwise, cost_v = vertical, cost_h = horizontal)
    focusmap = focusmap.reshape(focus.shape[1:])
    np.save(os.path.join(results_path, dataset, 'focusmap.npy'), focusmap)

    focusmap = np.load(os.path.join(results_path, dataset, 'focusmap.npy'))

    # get focused image
    focused = np.zeros(imgs[0].shape)
    for i in range(focusmap.shape[0]):
        for j in range(focusmap.shape[1]):
            focused[i,j,:] = imgs[focusmap[i][j]][i,j,:]

    # save images retrieved with graphcut
    plt.imshow(focusmap)
    plt.title('focusmap_%s' % (dataset))
    plt.colorbar()
    plt.savefig(os.path.join(results_path, dataset, 'focusmap.png'))
    plt.show()
    cv2.imwrite(os.path.join(results_path, dataset, '02focused_graphcut.png'), focused)

    # apply weighted median filter
    print('weighted median filter')
    filtered = cv2.ximgproc.weightedMedianFilter(np.argmax(focus, axis=0).astype(np.uint8), focusmap.astype(np.uint8), 7)

    # get focued image
    for i in range(filtered.shape[0]):
        for j in range(filtered.shape[1]):
            focused[i,j,:] = imgs[filtered[i,j]][i,j,:]

    # save images retrieved after weighted median filter
    plt.imshow(filtered)
    plt.title('filtered_focusmap_%s' % (dataset))
    plt.colorbar()
    plt.savefig(os.path.join(results_path, dataset, 'filtered.png'))
    plt.show()
    cv2.imwrite(os.path.join(results_path, dataset, '03focused_filtered.png'), focused)


if __name__=='__main__':
    images_path = 'dataset/images'
    datasets = ['boxes', 'cotton']

    for dataset in datasets:
        print('Start working on %s'%(dataset))

        # TASK1: Image Alignment
        print('TASK 1, Image Alignment')
        aligned_path = 'dataset/aligned'
        if not os.path.isdir(aligned_path):
            os.makedirs(aligned_path)
        align(images_path, dataset, aligned_path)
        
        # # TASK2: Initial depth from focus measure
        # # TASK3: All-in-focus image
        # print('TASK 2&3, Initial depth from focus measure & All-in-focus image')
        # results_path = 'dataset/focus'
        # if not os.path.isdir(results_path):
        #     os.makedirs(results_path)
        # depth_focus(aligned_path, dataset, results_path)
        #
        # # TASK4: Graph-cuts and weighted median filter
        # print('TASK4: Graph-cuts and weighted median filter')
        # graph_cut(aligned_path, dataset, results_path)
        #
        #



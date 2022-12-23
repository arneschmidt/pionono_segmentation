import os
import imageio
import errno
import csv
import torch
import mlflow
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import save_image
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import utils.globals as globals
import matplotlib.colors as mcolors

# for classes: NC, GG3, GG4, GG5, background
CLASS_COLORS_BGR = [[128, 255, 96], [32, 224, 255], [0, 104, 255], [0, 0, 255], [255, 255, 255]]

def save_model(model):
    model_dir = 'models'
    dir = os.path.join(globals.config['logging']['experiment_folder'], model_dir)
    os.makedirs(dir, exist_ok=True)
    out_path = os.path.join(dir, 'best_model.pth')
    torch.save(model, out_path)
    print('Best Model saved!')


def save_test_images(test_imgs:torch.Tensor, test_preds: np.array, test_labels: np.array, test_name: np.array, mode: str):
    visual_dir = 'qualitative_results/' + mode
    dir = os.path.join(globals.config['logging']['experiment_epoch_folder'], visual_dir)
    os.makedirs(dir, exist_ok=True)

    if len(test_labels.shape) == 3:
        test_labels = test_labels[0]

    test_preds = np.asarray(test_preds, dtype=np.uint8)
    test_labels = np.asarray(test_labels, dtype=np.uint8)

    # print("test name ", test_name)
    out_path = os.path.join(dir, 'img_' + test_name)
    save_image(test_imgs, out_path)

    test_pred_rgb = convert_classes_to_rgb(test_preds)
    out_path = os.path.join(dir, 'pred_' + test_name)
    imageio.imsave(out_path, test_pred_rgb)

    test_label_rgb = convert_classes_to_rgb(test_labels)
    out_path = os.path.join(dir, 'gt_' + test_name)
    imageio.imsave(out_path, test_label_rgb)


def save_test_image_variability(model, test_name, k, mode):
    no_samples_per_annotator = 6
    annotators = globals.config['data']['train']['masks']
    method = globals.config['model']['method']
    class_no = globals.config['data']['class_no']
    visual_dir = 'qualitative_results/' + mode
    dir = os.path.join(globals.config['logging']['experiment_epoch_folder'], visual_dir)
    dir = os.path.join(dir, 'variability')
    os.makedirs(dir, exist_ok=True)
    if method == 'pionono':
        for i in range(len(annotators)):
            a = annotators[i]
            a_dir = os.path.join(dir, a)
            os.makedirs(a_dir, exist_ok=True)
            annotator = torch.ones(model.unet_features.shape[0]) * i
            mean_pred = model.sample(use_z_mean=True, annotator=annotator)
            _, mean_pred = torch.max(mean_pred[:, 0:class_no], dim=1)
            mean_pred_k = convert_classes_to_rgb(mean_pred[k].cpu().detach().numpy())
            out_path = os.path.join(a_dir, 'pred_' + test_name[k].replace(".png", "_mean" + ".png"))
            imageio.imsave(out_path, mean_pred_k)
            for s in range(no_samples_per_annotator -1):
                pred = model.sample(use_z_mean=False, annotator=annotator)
                _, pred = torch.max(pred[:, 0:class_no], dim=1)
                out_path = os.path.join(a_dir, 'pred_' + test_name[k].replace(".png", "_s_" + str(s) + ".png"))
                pred_k = convert_classes_to_rgb(pred[k].cpu().detach().numpy())
                imageio.imsave(out_path, pred_k)

def save_model_distributions(model):
    dir_name = 'distributions'
    dir_path = os.path.join(globals.config['logging']['experiment_epoch_folder'], dir_name)
    os.makedirs(dir_path, exist_ok=True)
    annotators = globals.config['data']['train']['masks']
    method = globals.config['model']['method']
    if method == 'pionono':
        mu = model.z.posterior_mu.cpu().detach().numpy()
        covtril = model.z.posterior_covtril.cpu().detach().numpy()
        cov = np.zeros_like(covtril)
        for i in range(len(annotators)):
            cov[i] = np.matmul(covtril[i], covtril[i].transpose())
            np.savetxt(os.path.join(dir_path, "mu_" + str(i) + ".csv" ), np.round(mu[i], 4), delimiter=",", fmt="%.3f")
            np.savetxt(os.path.join(dir_path, "cov_" + str(i) + ".csv" ), np.round(cov[i], 4) , delimiter=",", fmt="%.3f")
        plot_and_save_distributions(mu, cov, dir_path)

def plot_and_save_distributions(mu_list, cov_list, dir_path):
    plt.figure()
    # plt.style.use('seaborn-dark')
    # plt.rcParams['figure.figsize'] = 14, 14
    no_annotators = mu_list.shape[0]

    twodim_mu_list = np.zeros(shape=[no_annotators, 2])
    twodim_cov_list = np.zeros(shape=[no_annotators, 2, 2])
    for i in range(no_annotators):
        twodim_mu_list[i] = mu_list[i][0:2]
        twodim_cov_list[i] = cov_list[i][0:2, 0:2]

    # Initializing the random seed
    random_seed = 0

    lim = np.max(np.abs(twodim_mu_list)) * 1.5 + np.max(np.abs(twodim_cov_list))
    x = np.linspace(- lim, lim, num=100)
    y = np.linspace(- lim, lim, num=100)
    X, Y = np.meshgrid(x, y)

    pdf_list = []

    for i in range(no_annotators):
        mean = twodim_mu_list[i]
        cov = twodim_cov_list[i]
        distr = multivariate_normal(cov=cov, mean=mean,
                                    seed=random_seed)
        pdf = np.zeros(X.shape)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                pdf[i, j] = distr.pdf([X[i, j], Y[i, j]])
        pdf_list.append(pdf)

   # Plotting contour plots
    annotators = globals.config['data']['train']['masks']
    colors = list(mcolors.TABLEAU_COLORS.keys())
    legend_list =[]
    for idx, val in enumerate(pdf_list):
        contourline = np.max(val) * (3/4)
        cntr = plt.contour(X, Y, val, levels=[contourline], colors=colors[idx], alpha=0.7)
        h, _ = cntr.legend_elements()
        legend_list.append(h[0])
    plt.legend(legend_list, annotators)
    plt.tight_layout()
    plt.savefig(os.path.join(dir_path, "dist_plot.jpg" ))
    plt.close()


def save_crowd_images(test_imgs:torch.Tensor, gt_pred: np.array, test_preds: np.array, test_labels: np.array, test_name: np.array, annotator, cm):
    visual_dir = 'qualitative_results/' + "train_crowd"
    dir = os.path.join(globals.config['logging']['experiment_epoch_folder'], visual_dir)
    os.makedirs(dir, exist_ok=True)

    test_preds = np.asarray(test_preds, dtype=np.uint8)
    test_labels = np.asarray(test_labels, dtype=np.uint8)

    # print("test name ", test_name)
    out_path = os.path.join(dir, 'img_' + test_name)
    save_image(test_imgs, out_path)

    test_pred_rgb = convert_classes_to_rgb(test_preds)
    out_path = os.path.join(dir, annotator + '_pred_' + test_name)
    imageio.imsave(out_path, test_pred_rgb)

    gt_pred_rgb = convert_classes_to_rgb(gt_pred)
    out_path = os.path.join(dir, 'gt_pred_' + test_name)
    imageio.imsave(out_path, gt_pred_rgb)

    test_label_rgb = convert_classes_to_rgb(test_labels)
    out_path = os.path.join(dir, annotator + '_gt_' + test_name)
    imageio.imsave(out_path, test_label_rgb)

    cm = cm.detach().cpu().numpy()
    plt.matshow(cm)
    out_path = os.path.join(dir, annotator + '_matrix_' + test_name)
    plt.savefig(out_path)
    plt.close()


def save_image_color_legend():
    # visual_dir = 'qualitative_results/'
    dir = globals.config['logging']['experiment_folder']
    os.makedirs(dir, exist_ok=True)
    class_no = globals.config['data']['class_no']
    class_names = globals.config['data']['class_names']

    fig = plt.figure()

    size = 100

    for class_id in range(class_no):
        # out_img[size*class_id:size*(class_id+1),:,:] = convert_classes_to_rgb(np.ones(size,size,3)*class_id, size,size)
        out_img = convert_classes_to_rgb(np.ones(shape=[size,size])*class_id)
        ax = fig.add_subplot(1, class_no, class_id+1)
        ax.imshow(out_img)
        ax.set_title(class_names[class_id])
        ax.axis('off')
    plt.savefig(os.path.join(dir, 'legend.png'))
    plt.close()


def convert_classes_to_rgb(seg_classes):
    h = seg_classes.shape[0]
    w = seg_classes.shape[1]
    seg_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    class_no = globals.config['data']['class_no']

    for class_id in range(class_no):
        # swap color channels because imageio saves images in RGB (not BGR)
        seg_rgb[:, :, 0][seg_classes == class_id] = CLASS_COLORS_BGR[class_id][2]
        seg_rgb[:, :, 1][seg_classes == class_id] = CLASS_COLORS_BGR[class_id][1]
        seg_rgb[:, :, 2][seg_classes == class_id] = CLASS_COLORS_BGR[class_id][0]

    return seg_rgb


def save_results(results):
    results_dir = 'quantitative_results'
    dir = os.path.join(globals.config['logging']['experiment_epoch_folder'], results_dir)
    os.makedirs(dir, exist_ok=True)
    out_path = os.path.join(dir, 'results.csv')

    with open(out_path, 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in results.items():
            writer.writerow([key, value])


def save_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    seg_model_ave_grads = []
    seg_model_max_grads = []
    seg_model_layers = []
    fcomb_ave_grads = []
    fcomb_max_grads = []
    fcomb_layers = []
    z_ave_grads = []
    z_max_grads = []
    z_layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n) and p.grad is not None:
            if 'seg_model' in n or 'unet' in n:
                seg_model_layers.append(n)
                seg_model_ave_grads.append(p.grad.abs().mean().cpu().detach().numpy())
                seg_model_max_grads.append(p.grad.abs().max().cpu().detach().numpy())
            elif 'head' in n or 'fcomb' in n:
                fcomb_layers.append(n)
                fcomb_ave_grads.append(p.grad.abs().mean().cpu().detach().numpy())
                fcomb_max_grads.append(p.grad.abs().max().cpu().detach().numpy())
            else:
                z_layers.append(n)
                z_ave_grads.append(p.grad.abs().mean().cpu().detach().numpy())
                z_max_grads.append(p.grad.abs().max().cpu().detach().numpy())
        # print(n)
    plot_gradients(seg_model_ave_grads, seg_model_max_grads, seg_model_layers, name='gradients_seg_model.jpg')
    plot_gradients(fcomb_ave_grads, fcomb_max_grads, fcomb_layers, name='gradients_fcomb.jpg')
    plot_gradients(z_ave_grads, z_max_grads, z_layers, name='gradients_z.jpg')

def plot_gradients(ave_grads, max_grads, layers, name):
    foldername = 'gradients'
    plt.figure()
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.5, lw=2, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.5, lw=2, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=3, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=-1, right=len(ave_grads))
    # plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([plt.Line2D([0], [0], color="c", lw=4),
                plt.Line2D([0], [0], color="b", lw=4),
                plt.Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    dir = os.path.join(globals.config['logging']['experiment_epoch_folder'], foldername)
    os.makedirs(dir, exist_ok=True)
    path = os.path.join(dir, name)
    plt.savefig(path)
    plt.close()

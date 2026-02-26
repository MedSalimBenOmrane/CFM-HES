import warnings
import os
import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

hemato_1 = (73, 119, 185.0)
eosin = (245, 143.0, 204)
safran = (248.0, 245, 152)
hemato_2 = (123, 153, 198)
Wgt = -np.log(np.array([hemato_1, eosin, safran, hemato_2]).T / 255)
warnings.filterwarnings("ignore")
softplus = torch.nn.Softplus()


def load_model(weights_path, model):
    weights = torch.load(weights_path)
    model.load_state_dict(weights)
    return model


def highlight_min(s):
    is_min = s == s.min()
    return ["color:green" if v else "" for v in is_min]


def highlight_max(s):
    is_max = s == s.max()
    return ["color:green" if v else "" for v in is_max]


def styley_df(df):
    subsets = ["MSE_H", "MSE_E", "MSE_S", "PieApp_H", "PieApp_E", "PieApp_S"]
    subsets1 = ["MSE", "PieApp"]
    styled_df = df.style.apply(highlight_min, subset=subsets).highlight_min(
        color="limegreen", subset=subsets1
    )
    styled_df = styled_df.highlight_max(color="limegreen", subset=["SSIM"]).apply(
        highlight_max, subset=["SSIM_H", "SSIM_E", "SSIM_S"]
    )
    return styled_df


def vectorize(im):
    V = -np.log(np.where(im == 0, 1, im) / 255.0)
    return V.transpose([2, 0, 1]).reshape(3, im.shape[0] * im.shape[1])


def unvectorize(V, N=500):
    return np.exp(-V.reshape((3, N, N)).transpose([1, 2, 0]))


def getImgs(p, data_path):
    liste = os.listdir(data_path + "IMG_" + str(p))
    liste.sort()
    E, HES, H, S = liste
    im = plt.imread(data_path + "IMG_" + str(p) + "/" + HES)
    im_H = plt.imread(data_path + "IMG_" + str(p) + "/" + H)
    im_E = plt.imread(data_path + "IMG_" + str(p) + "/" + E)
    im_S = plt.imread(data_path + "IMG_" + str(p) + "/" + S)
    im_H, im_E, im_S = (
        (im_H / 255.0).astype(np.float64),
        (im_E / 255.0).astype(np.float64),
        (im_S / 255.0).astype(np.float64),
    )
    return im, im_H, im_E, im_S

def getImgsGT(p, data_path):
    liste = os.listdir(data_path + "GTs_" + str(p))
    liste.sort()
    E, HES,_, H, S = liste
    im = plt.imread(data_path + "GTs_" + str(p) + "/" + HES)
    im_H = plt.imread(data_path + "GTs_" + str(p) + "/" + H)
    im_E = plt.imread(data_path + "GTs_" + str(p) + "/" + E)
    im_S = plt.imread(data_path + "GTs_" + str(p) + "/" + S)
    im_H, im_E, im_S = (
        (im_H / 255.0).astype(np.float64),
        (im_E / 255.0).astype(np.float64),
        (im_S / 255.0).astype(np.float64),
    )
    return im, im_H, im_E, im_S


def gen_HES(W, H_rec, N=500, M=500, device="cuda", BS=1, poids=[1.0, 1.0]):
    c11 = torch.matmul(W[:, 0].unsqueeze(1), H_rec[:, 0, :].unsqueeze(1))
    c12 = torch.matmul(W[:, 3].unsqueeze(1), H_rec[:, 3, :].unsqueeze(1))
    im_c1 = torch.exp(-(poids[0] * c11 + poids[1] * c12).reshape(BS, 3, N, M)).to(
        device
    )
    c2 = torch.matmul(W[:, 1].unsqueeze(1), H_rec[:, 1, :].unsqueeze(1))
    im_c2 = torch.exp(-c2.reshape(BS, 3, N, M)).to(device)
    c3 = torch.matmul(W[:, 2].unsqueeze(1), H_rec[:, 2, :].unsqueeze(1))
    im_c3 = torch.exp(-c3.reshape(BS, 3, N, M)).to(device)
    return im_c1, im_c2, im_c3


# def gen_HES_npy(W, H, poids=[1.0, 1.0]):
#     c11 = np.kron(W[:, 0, np.newaxis], np.transpose(H[0, :, np.newaxis]))
#     c12 = np.kron(W[:, 3, np.newaxis], np.transpose(H[3, :, np.newaxis]))
#     c2 = np.kron(W[:, 1, np.newaxis], np.transpose(H[1, :, np.newaxis]))
#     c3 = np.kron(W[:, 2, np.newaxis], np.transpose(H[2, :, np.newaxis]))
#     im_c1 = (
#         np.exp(-(poids[0] * c11 + poids[1] * c12))
#         .reshape(3, 500, 500)
#         .transpose([1, 2, 0])
#     )
#     im_c2 = np.exp(-c2).reshape(3, 500, 500).transpose([1, 2, 0])
#     im_c3 = np.exp(-c3).reshape(3, 500, 500).transpose([1, 2, 0])
#     return im_c1, im_c2, im_c3


# def gen_HESbis_npy(W, H, poids=[1.0, 1.0]):
#     c11 = np.kron(W[:, 0, np.newaxis], np.transpose(H[0, :, np.newaxis]))
#     c12 = np.kron(W[:, 3, np.newaxis], np.transpose(H[3, :, np.newaxis]))
#     c2 = np.kron(W[:, 1, np.newaxis], np.transpose(H[1, :, np.newaxis]))
#     c3 = np.kron(W[:, 2, np.newaxis], np.transpose(H[2, :, np.newaxis]))
#     im_c1 = (
#         np.exp(-(poids[0] * c11 + poids[1] * c12))
#         .reshape(3, 500, 500)
#         .transpose([1, 2, 0])
#     )
#     im_c2 = np.exp(-c2).reshape(3, 500, 500).transpose([1, 2, 0])
#     im_c3 = np.exp(-c3).reshape(3, 500, 500).transpose([1, 2, 0])
#     return im_c1, im_c2, im_c3


# def gen_HEStri_npy(W, H, poids=[1.0, 1.0]):
#     c1 = np.kron(W[:, 0, np.newaxis], np.transpose(H[0, :, np.newaxis]))
#     c22 = np.kron(W[:, 3, np.newaxis], np.transpose(H[3, :, np.newaxis]))
#     c21 = np.kron(W[:, 1, np.newaxis], np.transpose(H[1, :, np.newaxis]))
#     c3 = np.kron(W[:, 2, np.newaxis], np.transpose(H[2, :, np.newaxis]))
#     im_c2 = (
#         np.exp(-(poids[0] * c21 + poids[1] * c22))
#         .reshape(3, 500, 500)
#         .transpose([1, 2, 0])
#     )
#     im_c1 = np.exp(-c1).reshape(3, 500, 500).transpose([1, 2, 0])
#     im_c3 = np.exp(-c3).reshape(3, 500, 500).transpose([1, 2, 0])
#     return im_c1, im_c2, im_c3


def plot_list_images(list_images, PREFIX, prefix, path_to_data, saving_path):
    for idx in range(10):
        key = list(list_images.keys())[idx]
        c1, c2, c3 = list_images[key]
        im, im_H, im_E, im_S = getImgs(key, path_to_data)
        _, ax = plt.subplots(ncols=4, nrows=2, figsize=(12, 6))
        ax[0, 0].imshow(im)
        ax[0, 0].set_title("IMG_" + str(key))
        ax[0, 1].imshow(c1)
        ax[0, 2].imshow(c2)
        ax[0, 3].imshow(c3)
        ax[1, 1].imshow(im_H)
        ax[1, 2].imshow(im_E)
        ax[1, 3].imshow(im_S)
        for i in range(2):
            for k in range(4):
                ax[i, k].axis(False)
        plt.tight_layout()
        plt.savefig(f"{saving_path}/{PREFIX}/IMG_{str(key)}_{prefix}.jpeg")


def training(
    W, H0, model, optimizer, train_loader, loss_fn, device, N=500, M=500
) -> float:
    train_loss = 0.0
    model.train()
    for V, _, im_H, im_E, im_S in tqdm(train_loader):
        optimizer.zero_grad()
        H_rec = model(V.to(device), W, H0)
        im_H, im_E, im_S = [
            e.to(device).requires_grad_(True) for e in [im_H, im_E, im_S]
        ]
        im_c1, im_c2, im_c3 = [
            e.requires_grad_(True)
            for e in gen_HES(W, H_rec, N, M, device, BS=V.shape[0])
        ]
        loss = (loss_fn(im_H, im_c1) + loss_fn(im_E, im_c2) + loss_fn(im_S, im_c3)) / 3
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        del im_c1, im_c2, im_c3, im_H, im_E, im_S, V, H_rec, loss
    return train_loss / len(train_loader)


def testing(W, H1, model, test_loader, loss_fn, device, N=500, M=500) -> float:
    test_loss = 0.0
    model.eval()
    with torch.no_grad():
        for V, _, im_H, im_E, im_S in tqdm(test_loader):
            H_rec = model(V.to(device), W, H1)
            im_H, im_E, im_S = [e.to(device) for e in [im_H, im_E, im_S]]
            im_c1, im_c2, im_c3 = gen_HES(W, H_rec, N, M, device, BS=V.shape[0])
            loss = (
                loss_fn(im_H, im_c1) + loss_fn(im_E, im_c2) + loss_fn(im_S, im_c3)
            ) / 3
            test_loss += loss.item()
            del im_c1, im_c2, im_c3, im_H, im_E, im_S, V, H_rec, loss
    return test_loss / len(test_loader)


def evaluation(
    W,
    H0,
    model,
    data_loader,
    device,
    mse_metric,
    ssim_metric,
    pieapp_metric,
    N=500,
    M=500,
):
    val_MSE, val_SSIM, val_PIEAPP = 0.0, 0.0, 0.0
    with torch.no_grad():
        for V, _, im_H, im_E, im_S in tqdm(data_loader):
            H_rec = model(V.to(device), W, H0)
            im_H, im_E, im_S = [e.to(device) for e in [im_H, im_E, im_S]]
            im_c1, im_c2, im_c3 = gen_HES(W, H_rec, N, M, device, BS=V.shape[0])
            val_MSE += (
                mse_metric(im_H, im_c1)
                + mse_metric(im_E, im_c2)
                + mse_metric(im_S, im_c3)
            ) / 3
            val_SSIM += (
                ssim_metric(im_H, im_c1)
                + ssim_metric(im_E, im_c2)
                + ssim_metric(im_S, im_c3)
            ) / 3
            val_PIEAPP += (
                pieapp_metric(im_H, im_c1)
                + pieapp_metric(im_E, im_c2)
                + pieapp_metric(im_S, im_c3)
            ) / 3
    return (
        val_MSE / len(data_loader),
        val_SSIM / len(data_loader),
        val_PIEAPP / len(data_loader),
    )


def generate_results(
    W,
    H1,
    model,
    df_test,
    test_loader,
    device,
    mse_metric,
    ssim_metric,
    pieapp_metric,
    N=500,
    M=500,
    poids=[1.0, 1.0],
):
    cols = [
        "image",
        "MSE_H",
        "MSE_E",
        "MSE_S",
        "MSE",
        "SSIM_H",
        "SSIM_E",
        "SSIM_S",
        "SSIM",
        "PieApp_H",
        "PieApp_E",
        "PieApp_S",
        "PieApp",
    ]
    df = pd.DataFrame(columns=cols, index=range(1 + len(df_test)))
    i = 0
    list_images = {}
    with torch.no_grad():
        for V, _, im_H, im_E, im_S in tqdm(test_loader):
            H_rec = model(V.to(device), W, H1)
            im_H, im_E, im_S = [e.to(device) for e in [im_H, im_E, im_S]]
            im_c1, im_c2, im_c3 = gen_HES(
                W, H_rec, N, M, device, BS=V.shape[0], poids=poids
            )
            for n in range(V.shape[0]):
                a, b, c = (
                    im_c1[n].unsqueeze(0),
                    im_c2[n].unsqueeze(0),
                    im_c3[n].unsqueeze(0),
                )
                d, e, f = (
                    im_H[n].unsqueeze(0),
                    im_E[n].unsqueeze(0),
                    im_S[n].unsqueeze(0),
                )
                list_images[str(df_test["image_index"].values[i])] = [
                    img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
                    for img in [a, b, c]
                ]
                mses = [
                    mse_metric(a, d).item(),
                    mse_metric(b, e).item(),
                    mse_metric(c, f).item(),
                ]
                ssims = [
                    ssim_metric(a, d).item(),
                    ssim_metric(b, e).item(),
                    ssim_metric(c, f).item(),
                ]
                pieapps = [
                    pieapp_metric(a, d).item(),
                    pieapp_metric(b, e).item(),
                    pieapp_metric(c, f).item(),
                ]
                df.iloc[i] = ["IMG_" + str(df_test["image_index"].values[i])] + mses + [np.mean(mses)] + ssims + [np.mean(ssims)] + pieapps + [np.mean(pieapps)]  # type: ignore
                i += 1
    df[list(df.columns)[1:]] = df[list(df.columns)[1:]].astype(float)
    df.iloc[i] = ["Average"] + list(df[list(df.columns)[1:]].mean())  # type: ignore
    return df, list_images


def compute_metric(metric, gts, recs):
    metrics = [metric(gts[i], recs[i]).item() for i in range(len(gts))]
    return metrics + [np.mean(metrics)]


def get_dfs_and_images(
    model, Wgt, df_test, test_loader, params, device, regu, MSE, SSIMLoss, PIEAPP
):
    N, M = 500, 500
    cols = [
        "image",
        "MSE_H",
        "MSE_E",
        "MSE_S",
        "MSE",
        "SSIM_H",
        "SSIM_E",
        "SSIM_S",
        "SSIM",
        "PieApp_H",
        "PieApp_E",
        "PieApp_S",
        "PieApp",
    ]
    df = pd.DataFrame(columns=cols, index=range(1 + len(df_test)))
    list_images = {}
    params = torch.tensor(params, dtype=torch.float32, device=device)
    Wgt = torch.tensor(Wgt, dtype=torch.float32, device=device)
    for i, (V, _, im_H, im_E, im_S) in enumerate(tqdm(test_loader)):
        V = V.to(device)
        H0 = (torch.pinverse(Wgt) @ V).clamp(min=0.0)
        if "TVLQ" in regu:
            H0 = H0.reshape(4, N, M).permute([1, 2, 0])
        H_opt = model(V, H0, params)
        c1, c2, c3 = gen_HES(Wgt, H_opt)
        im_H, im_E, im_S = im_H.to(device), im_E.to(device), im_S.to(device)
        mses = compute_metric(MSE, [im_H, im_E, im_S], [c1, c2, c3])
        ssims = compute_metric(SSIMLoss, [im_H, im_E, im_S], [c1, c2, c3])
        pieapps = compute_metric(PIEAPP, [im_H, im_E, im_S], [c1, c2, c3])
        im_c1 = c1.squeeze(0).permute(1, 2, 0).cpu().numpy()
        im_c2 = c2.squeeze(0).permute(1, 2, 0).cpu().numpy()
        im_c3 = c3.squeeze(0).permute(1, 2, 0).cpu().numpy()
        list_images[str(df_test["image_index"].values[i])] = [im_c1, im_c2, im_c3]
        name = "IMG_" + str(df_test["image_index"].values[i])
        df.iloc[i] = [name] + mses + ssims + pieapps  # type: ignore
    df[list(df.columns)[1:]] = df[list(df.columns)[1:]].astype(float)
    df.iloc[i + 1] = ["Average"] + list(df[list(df.columns)[1:]].mean())  # type: ignore
    return df, list_images


# def generate_results_nelder(
#     Wgt,
#     params,
#     model,
#     df_test,
#     test_loader,
#     device,
#     mse_metric,
#     ssim_metric,
#     pieapp_metric,
#     N=500,
#     M=500,
#     S=4,
#     poids=[1.0, 1.0],
# ):
#     cols = [
#         "image",
#         "MSE_H",
#         "MSE_E",
#         "MSE_S",
#         "MSE",
#         "SSIM_H",
#         "SSIM_E",
#         "SSIM_S",
#         "SSIM",
#         "PieApp_H",
#         "PieApp_E",
#         "PieApp_S",
#         "PieApp",
#     ]
#     df = pd.DataFrame(columns=cols, index=range(1 + len(df_test)))
#     params = torch.tensor(params, dtype=torch.float32, device=device)
#     i = 0
#     list_images = {}
#     with torch.no_grad():
#         for V, _, im_H, im_E, im_S in tqdm(test_loader):
#             V = V.squeeze(1).squeeze(0).type(torch.float32).to(device)
#             H0 = (torch.pinverse(Wgt) @ V).clamp(min=0.0).reshape(N, M, S)
#             im_H, im_E, im_S = im_H.to(device), im_E.to(device), im_S.to(device)
#             H1 = model(V, H0, params)
#             c1, c2, c3 = gen_HES(Wgt, H1.unsqueeze(0), poids=poids)
#             mses = [
#                 mse_metric(im_H, c1).item(),
#                 mse_metric(im_E, c2).item(),
#                 mse_metric(im_S, c3).item(),
#             ]
#             ssims = [
#                 ssim_metric(im_H, c1).item(),
#                 ssim_metric(im_E, c2).item(),
#                 ssim_metric(im_S, c3).item(),
#             ]
#             pieapps = [
#                 pieapp_metric(im_H, c1).item(),
#                 pieapp_metric(im_E, c2).item(),
#                 pieapp_metric(im_S, c3).item(),
#             ]
#             imgs = [
#                 c1.squeeze(0).permute(1, 2, 0).detach().cpu().numpy(),
#                 c2.squeeze(0).permute(1, 2, 0).detach().cpu().numpy(),
#                 c3.squeeze(0).permute(1, 2, 0).detach().cpu().numpy(),
#             ]
#             list_images[str(df_test["image_index"].values[i])] = imgs
#             df.iloc[i] = ["IMG_" + str(df_test["image_index"].values[i])] + mses + [np.mean(mses)] + ssims + [np.mean(ssims)] + pieapps + [np.mean(pieapps)]  # type: ignore
#             i += 1
#     df[list(df.columns)[1:]] = df[list(df.columns)[1:]].astype(float)
#     df.iloc[i] = ["Average"] + list(df[list(df.columns)[1:]].mean())  # type: ignore
#     return df, list_images
